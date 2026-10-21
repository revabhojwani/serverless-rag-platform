import base64
import binascii
import os
import re

import numpy as np

import common


def handler(event, context):
    tenant_id = common.authenticate(event)
    if not tenant_id:
        return common.response(403, {"error": "invalid tenant credentials"})

    try:
        body = common.parse_body(event)
        filename = os.path.basename(body["filename"])
        pdf_bytes = base64.b64decode(body["content_base64"], validate=True)
    except (KeyError, ValueError, binascii.Error):
        return common.response(400, {"error": "body needs filename and content_base64"})
    if not re.fullmatch(r"[\w .()-]+\.pdf", filename, re.IGNORECASE):
        return common.response(400, {"error": "only .pdf filenames are accepted"})

    text = common.extract_pdf_text(pdf_bytes)
    pieces = common.chunk_text(text)
    if not pieces:
        return common.response(422, {"error": "no extractable text in the PDF"})

    common.s3().put_object(Bucket=common.BUCKET, Key=f"{tenant_id}/docs/{filename}", Body=pdf_bytes)

    new_vecs = common.embed(pieces)
    vecs, chunks = common.load_index(tenant_id)
    # re-uploading a file replaces its old chunks
    keep = [i for i, c in enumerate(chunks) if c["source"] != filename]
    vecs, chunks = vecs[keep], [chunks[i] for i in keep]
    chunks += [{"source": filename, "text": p} for p in pieces]
    vecs = np.vstack([vecs, new_vecs]) if len(vecs) else new_vecs
    common.save_index(tenant_id, vecs, chunks)

    return common.response(200, {"tenant_id": tenant_id, "filename": filename, "chunks_added": len(pieces)})
