import urllib.error

import common

TOP_K = 4


def handler(event, context):
    tenant_id = common.authenticate(event)
    if not tenant_id:
        return common.response(403, {"error": "invalid tenant credentials"})

    try:
        question = common.parse_body(event)["question"].strip()
    except (KeyError, ValueError, AttributeError):
        return common.response(400, {"error": "body needs a question"})
    if not question:
        return common.response(400, {"error": "question is empty"})

    vecs, chunks = common.load_index(tenant_id)
    if not chunks:
        return common.response(404, {"error": "no documents uploaded for this tenant yet"})

    scores = vecs @ common.embed([question])[0]
    top = scores.argsort()[::-1][:TOP_K]
    contexts = [chunks[i] for i in top]

    try:
        answer = common.call_groq(question, contexts)
    except (urllib.error.URLError, KeyError, RuntimeError) as exc:
        return common.response(502, {"error": f"LLM call failed: {exc}"})

    sources = sorted({c["source"] for c in contexts})
    common.log_chat(tenant_id, question, answer, sources)
    return common.response(200, {"tenant_id": tenant_id, "answer": answer, "sources": sources})
