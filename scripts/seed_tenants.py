"""Creates the two demo tenants in DynamoDB and prints their keys once.

Usage: python scripts/seed_tenants.py <table-name> [region]
Needs AWS credentials configured locally (aws configure).
"""
import hashlib
import secrets
import sys

import boto3

TENANTS = {"student-portal": "Student Portal", "faculty-portal": "Faculty Portal"}


def main():
    table_name = sys.argv[1]
    region = sys.argv[2] if len(sys.argv) > 2 else None
    table = boto3.resource("dynamodb", region_name=region).Table(table_name)
    for tenant_id, name in TENANTS.items():
        key = secrets.token_urlsafe(24)
        table.put_item(
            Item={
                "tenant_id": tenant_id,
                "sk": "META",
                "name": name,
                "key_hash": hashlib.sha256(key.encode()).hexdigest(),
            }
        )
        print(f"{tenant_id}  key: {key}")
    print("Save these keys now. Only their hashes are stored.")


if __name__ == "__main__":
    main()
