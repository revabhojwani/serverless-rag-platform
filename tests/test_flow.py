import base64
import hashlib
import io
import json
import os
import re
import sys

import boto3
import numpy as np
import pytest
from moto import mock_aws
from reportlab.pdfgen import canvas

os.environ.update(
    AWS_ACCESS_KEY_ID="x", AWS_SECRET_ACCESS_KEY="x", AWS_DEFAULT_REGION="us-east-1",
    BUCKET="test-bucket", TABLE="test-table", GROQ_API_KEY="fake",
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import common  # noqa: E402
import ingest  # noqa: E402
import query  # noqa: E402


def fake_embed(texts):
    """Hashed bag-of-words, 384 dims, normalised. Stands in for the real model, which needs a download."""
    out = np.zeros((len(texts), 384), dtype="float32")
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 384] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return out / norms


def make_pdf(lines):
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for line in lines:
        c.drawString(50, y, line)
        y -= 20
    c.save()
    return buf.getvalue()


def event(tenant, key, body):
    return {"headers": {"X-Tenant-Id": tenant, "X-Tenant-Key": key}, "body": json.dumps(body)}


@pytest.fixture
def aws(monkeypatch):
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="test-bucket")
        ddb = boto3.resource("dynamodb")
        ddb.create_table(
            TableName="test-table",
            KeySchema=[{"AttributeName": "tenant_id", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}],
            AttributeDefinitions=[{"AttributeName": "tenant_id", "AttributeType": "S"}, {"AttributeName": "sk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        for t in ("student-portal", "faculty-portal"):
            ddb.Table("test-table").put_item(Item={"tenant_id": t, "sk": "META", "key_hash": common.hash_key(f"{t}-key")})
        common._s3 = None
        common._table = None
        monkeypatch.setattr(common, "embed", fake_embed)
        monkeypatch.setattr(common, "call_groq", lambda q, ctx: "ANSWER FROM: " + " | ".join(c["text"] for c in ctx))
        yield


def upload(tenant, name, lines):
    body = {"filename": name, "content_base64": base64.b64encode(make_pdf(lines)).decode()}
    return ingest.handler(event(tenant, f"{tenant}-key", body), None)


def test_full_flow_and_isolation(aws):
    r = upload("student-portal", "handbook.pdf", ["Minimum attendance for exams is seventy five percent.", "Hostel curfew is ten pm."])
    assert r["statusCode"] == 200 and json.loads(r["body"])["chunks_added"] >= 1
    r = upload("faculty-portal", "policy.pdf", ["Faculty leave requests need approval from the head of department."])
    assert r["statusCode"] == 200

    r = query.handler(event("student-portal", "student-portal-key", {"question": "minimum attendance for exams"}), None)
    body = json.loads(r["body"])
    assert r["statusCode"] == 200
    assert "seventy five percent" in body["answer"]
    assert body["sources"] == ["handbook.pdf"]
    assert "leave requests" not in body["answer"]

    r = query.handler(event("faculty-portal", "faculty-portal-key", {"question": "minimum attendance for exams"}), None)
    body = json.loads(r["body"])
    assert body["sources"] == ["policy.pdf"]
    assert "seventy five" not in body["answer"]

    keys = [o["Key"] for o in boto3.client("s3").list_objects_v2(Bucket="test-bucket")["Contents"]]
    assert all(k.startswith(("student-portal/", "faculty-portal/")) for k in keys)

    chats = common.table().query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("tenant_id").eq("student-portal") & boto3.dynamodb.conditions.Key("sk").begins_with("CHAT#")
    )["Items"]
    assert len(chats) == 1 and chats[0]["question"] == "minimum attendance for exams"


def test_auth_rejected(aws):
    body = {"filename": "a.pdf", "content_base64": base64.b64encode(make_pdf(["hello"])).decode()}
    assert ingest.handler(event("student-portal", "wrong", body), None)["statusCode"] == 403
    assert ingest.handler(event("student-portal", "faculty-portal-key", body), None)["statusCode"] == 403
    assert ingest.handler(event("nope", "x", body), None)["statusCode"] == 403
    assert query.handler(event("student-portal", "wrong", {"question": "hi"}), None)["statusCode"] == 403


def test_bad_inputs(aws):
    assert query.handler(event("student-portal", "student-portal-key", {"question": "hi"}), None)["statusCode"] == 404
    assert ingest.handler(event("student-portal", "student-portal-key", {"filename": "x.txt", "content_base64": "aGk="}), None)["statusCode"] == 400
    assert ingest.handler(event("student-portal", "student-portal-key", {"nope": 1}), None)["statusCode"] == 400


def test_reupload_replaces_chunks(aws):
    upload("student-portal", "handbook.pdf", ["Old rule about parking."])
    upload("student-portal", "handbook.pdf", ["New rule about libraries."])
    _, chunks = common.load_index("student-portal")
    assert len(chunks) == 1 and "libraries" in chunks[0]["text"]


def test_embed_pooling_and_normalisation(monkeypatch):
    """Checks mean pooling ignores padding and output is unit length. Real model files are not needed."""

    class Enc:
        def __init__(self, n_real, n_total):
            self.ids = [5] * n_real + [0] * (n_total - n_real)
            self.attention_mask = [1] * n_real + [0] * (n_total - n_real)

    class FakeTok:
        def encode_batch(self, texts):
            return [Enc(len(t.split()), 6) for t in texts]

    class FakeSess:
        def get_inputs(self):
            return [type("I", (), {"name": n}) for n in ("input_ids", "attention_mask")]

        def run(self, _, feeds):
            assert "token_type_ids" not in feeds
            b, s = feeds["input_ids"].shape
            hidden = np.ones((b, s, 384), dtype="float32")
            hidden[:, 3:, :] = 1000.0  # padding positions must not leak into the mean
            return [hidden]

    monkeypatch.setattr(common, "_model", (FakeTok(), FakeSess()))
    out = common.embed(["a b c", "a b"])
    assert out.shape == (2, 384)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert np.allclose(out[0], out[1])
