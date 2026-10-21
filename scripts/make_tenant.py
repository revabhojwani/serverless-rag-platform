"""Prints a DynamoDB item for a tenant plus its secret key. No AWS credentials needed.

  python scripts/make_tenant.py student-portal "Student Portal"
  python scripts/make_tenant.py faculty-portal "Faculty Portal"

In the DynamoDB console: your table > Explore table items > Create item > JSON view
(turn on "View DynamoDB JSON"), paste the item, and save. Keep the printed key; only its hash is stored.
"""
import hashlib
import json
import re
import secrets
import sys

tenant_id, name = sys.argv[1], sys.argv[2]
if not re.fullmatch(r"[a-z0-9-]{1,40}", tenant_id):
    sys.exit("tenant id must be lowercase letters, digits and hyphens")

key = secrets.token_urlsafe(24)
item = {
    "tenant_id": {"S": tenant_id},
    "sk": {"S": "META"},
    "name": {"S": name},
    "key_hash": {"S": hashlib.sha256(key.encode()).hexdigest()},
}
print(f"\nTENANT KEY (save it now): {key}\n")
print("DynamoDB item to paste:\n")
print(json.dumps(item, indent=2))
