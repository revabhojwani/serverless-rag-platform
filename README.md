# Serverless RAG-as-a-Service

Two Lambda functions behind an HTTP API. Tenants authenticate with `x-tenant-id` and `x-tenant-key` headers.

    POST /ingest   {"filename": "rules.pdf", "content_base64": "..."}   -> chunk, embed, store index in S3
    POST /query    {"question": "..."}                                  -> retrieve top 4 chunks, answer with Groq

Storage layout
- S3: `<tenant_id>/docs/<file>.pdf`, `<tenant_id>/index/vectors.npy`, `<tenant_id>/index/chunks.json`
- DynamoDB, one table, partition key `tenant_id`, sort key `sk`: item `META` holds the tenant name and key hash, items `CHAT#<timestamp>` hold query history

## Deploy (AWS console, one region, e.g. ap-southest-2)

1. Build the package: `python build.py` (or `bash build.sh` on Mac/Linux). It prints the unzipped size. It must be under 250 MB.
2. S3: create a bucket, e.g. `rag-platform-<yourname>`. Upload `lambda.zip` into it (zips over 50 MB cannot be uploaded directly in the Lambda console).
3. DynamoDB: create a table, e.g. `rag-platform`, partition key `tenant_id` (String), sort key `sk` (String), on-demand capacity.
4. IAM: create a role for Lambda with the `AWSLambdaBasicExecutionRole` policy plus an inline policy allowing `s3:GetObject`, `s3:PutObject` on `arn:aws:s3:::<bucket>/*`, `s3:ListBucket` on `arn:aws:s3:::<bucket>` (without it S3 answers a missing index file with access denied instead of not found, and the first upload for a tenant fails), and `dynamodb:GetItem`, `dynamodb:PutItem` on the table.
5. Lambda: create two functions, runtime Python 3.12, x86_64, that role, "Upload from Amazon S3 location" pointing at the zip.
   - `rag-ingest`, handler `ingest.handler`
   - `rag-query`, handler `query.handler`
   - Both: memory 1024 MB, timeout 60 s, environment variables `BUCKET`, `TABLE`. On `rag-query` also set `GROQ_API_KEY`.
6. API Gateway: create an HTTP API, add routes `POST /ingest` to `rag-ingest` and `POST /query` to `rag-query`. Copy the invoke URL.
7. Create tenants without AWS keys on your laptop: run `python scripts/make_tenant.py student-portal "Student Portal"` and the same for `faculty-portal "Faculty Portal"`. Paste each printed item into the DynamoDB console (Explore table items, Create item, JSON view) and save the printed keys. `scripts/seed_tenants.py` does the same from the command line if you have AWS credentials configured.
8. Test:

       python scripts/client.py <invoke-url> student-portal <key> upload handbook.pdf
       python scripts/client.py <invoke-url> student-portal <key> ask "What is the minimum attendance?"
       python scripts/client.py <invoke-url> faculty-portal <key> ask "What is the minimum attendance?"

   The last call should say the answer is not in the documents, since that tenant has no handbook. That is the isolation demo.

Budget: Lambda, API Gateway, S3 and DynamoDB at this volume stay inside the free tier. Set a budget alert in AWS Billing before you start. Groq's free tier covers the LLM calls.

## Known risks

- Embeddings: the zip cannot hold PyTorch, so `all-MiniLM-L6-v2` runs as a quantized ONNX file through ONNX Runtime. Same model and 384-dim vectors as sentence-transformers, about 172 MB unzipped.
- The embedding step has only been tested with a stand-in model here, because Hugging Face is blocked in my sandbox. The first real run on your machine is the test. `build.sh` downloads the model, so run it before anything else.
- Cold start: the first call after idle loads the model and takes a few seconds. Later calls are fast.
- Groq model name: the default is `openai/gpt-oss-120b`. Groq shut down `llama-3.3-70b-versatile` on 16 Aug 2026. If Groq retires the default too, set the `GROQ_MODEL` environment variable on `rag-query` to a current model from console.groq.com/docs/models. A Groq error shows up in the 502 response body.
- The index is one file per tenant, rewritten on every upload. Two simultaneous uploads for the same tenant can overwrite each other. Fine for the demo.
