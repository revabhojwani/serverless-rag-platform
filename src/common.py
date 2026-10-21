import hashlib
import hmac
import io
import json
import os
import re
import time
import urllib.error
import urllib.request

import boto3
import numpy as np

BUCKET = os.environ.get("BUCKET", "")
TABLE = os.environ.get("TABLE", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MAX_TOKENS = 256

CHUNK_WORDS = 150
CHUNK_OVERLAP = 30

_s3 = None
_table = None
_model = None


def s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


def table():
    global _table
    if _table is None:
        _table = boto3.resource("dynamodb").Table(TABLE)
    return _table


def response(status, body):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def hash_key(key):
    return hashlib.sha256(key.encode()).hexdigest()


def authenticate(event):
    """Return the tenant_id if the x-tenant-id / x-tenant-key headers match a tenant in DynamoDB, else None."""
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    tenant_id = headers.get("x-tenant-id", "").strip()
    key = headers.get("x-tenant-key", "").strip()
    if not tenant_id or not key or not re.fullmatch(r"[a-z0-9-]{1,40}", tenant_id):
        return None
    item = table().get_item(Key={"tenant_id": tenant_id, "sk": "META"}).get("Item")
    if not item:
        return None
    if not hmac.compare_digest(item["key_hash"], hash_key(key)):
        return None
    return tenant_id


def parse_body(event):
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64

        raw = base64.b64decode(raw).decode()
    return json.loads(raw)


def extract_pdf_text(pdf_bytes):
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def chunk_text(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks, step = [], size - overlap
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def _get_model():
    """Load the bundled all-MiniLM-L6-v2 (quantized ONNX) and its tokenizer once per container."""
    global _model
    if _model is None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        tok.enable_truncation(max_length=MAX_TOKENS)
        tok.enable_padding()
        sess = ort.InferenceSession(os.path.join(MODEL_DIR, "model.onnx"), providers=["CPUExecutionProvider"])
        _model = (tok, sess)
    return _model


def embed(texts, batch_size=16):
    """Mean-pooled, L2-normalised float32 embeddings, shape (len(texts), 384)."""
    tok, sess = _get_model()
    wanted = {i.name for i in sess.get_inputs()}
    pooled = []
    for start in range(0, len(texts), batch_size):
        enc = tok.encode_batch(texts[start:start + batch_size])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        hidden = sess.run(None, {k: v for k, v in feeds.items() if k in wanted})[0]
        m = mask[..., None].astype("float32")
        pooled.append((hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None))
    vecs = np.vstack(pooled).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vecs / norms


def _index_keys(tenant_id):
    return f"{tenant_id}/index/vectors.npy", f"{tenant_id}/index/chunks.json"


def load_index(tenant_id):
    vec_key, chunk_key = _index_keys(tenant_id)
    try:
        vecs = np.load(io.BytesIO(s3().get_object(Bucket=BUCKET, Key=vec_key)["Body"].read()))
        chunks = json.loads(s3().get_object(Bucket=BUCKET, Key=chunk_key)["Body"].read())
        return vecs, chunks
    except s3().exceptions.NoSuchKey:
        return np.zeros((0, 384), dtype="float32"), []


def save_index(tenant_id, vecs, chunks):
    vec_key, chunk_key = _index_keys(tenant_id)
    buf = io.BytesIO()
    np.save(buf, vecs.astype("float32"))
    s3().put_object(Bucket=BUCKET, Key=vec_key, Body=buf.getvalue())
    s3().put_object(Bucket=BUCKET, Key=chunk_key, Body=json.dumps(chunks).encode())


def call_groq(question, contexts):
    context_block = "\n\n".join(f"[{i + 1}] ({c['source']}) {c['text']}" for i, c in enumerate(contexts))
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question using only the numbered context passages. "
                "Cite passages like [1]. If the context does not contain the answer, "
                "say you could not find it in the documents."
            ),
        },
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
    ]
    payload = json.dumps({"model": GROQ_MODEL, "messages": messages, "temperature": 0.1}).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "User-Agent": "rag-platform/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"Groq HTTP {err.code}: {err.read().decode()[:300]}") from err


def log_chat(tenant_id, question, answer, sources):
    table().put_item(
        Item={
            "tenant_id": tenant_id,
            "sk": f"CHAT#{int(time.time() * 1000)}",
            "question": question,
            "answer": answer,
            "sources": sources,
        }
    )
