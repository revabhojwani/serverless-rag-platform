"""Small CLI for the demo.

  python scripts/client.py <api-url> <tenant-id> <tenant-key> upload path/to/file.pdf
  python scripts/client.py <api-url> <tenant-id> <tenant-key> ask What is the attendance rule?

<api-url> is the HTTP API invoke URL without a trailing slash.
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request


def post(url, tenant_id, key, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-tenant-id": tenant_id, "x-tenant-key": key},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.dumps(json.loads(resp.read()), indent=2)
    except urllib.error.HTTPError as err:
        return f"HTTP {err.code}: {err.read().decode()}"


def main():
    api, tenant_id, key, action = sys.argv[1:5]
    # quotes are optional: everything after the action is the question or the file path
    arg = " ".join(sys.argv[5:]).strip(" \"'\u201c\u201d\u2018\u2019")
    if action == "upload":
        with open(arg, "rb") as f:
            payload = {"filename": os.path.basename(arg), "content_base64": base64.b64encode(f.read()).decode()}
        print(post(f"{api}/ingest", tenant_id, key, payload))
    elif action == "ask":
        print(post(f"{api}/query", tenant_id, key, {"question": arg}))
    else:
        sys.exit("action must be upload or ask")


if __name__ == "__main__":
    main()
