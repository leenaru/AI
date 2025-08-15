import boto3, hashlib, time
from server.core.config import settings

s3cfg = settings.cfg.get("stores", {}).get("s3", {})
s3 = boto3.client(
    "s3",
    endpoint_url=s3cfg.get("endpoint_url"),
    aws_access_key_id=s3cfg.get("access_key"),
    aws_secret_access_key=s3cfg.get("secret_key"),
    region_name="us-east-1",
    use_ssl=s3cfg.get("secure", False),
)
BUCKET = s3cfg.get("bucket", "ai-vision-temp")

def persist_temp_artifacts(image_bytes: bytes, meta: dict):
    if not image_bytes:
        return None
    h = hashlib.sha1(image_bytes).hexdigest()
    ts = int(time.time())
    key = f"temp/{ts}/{h}.bin"
    s3.put_object(Bucket=BUCKET, Key=key, Body=image_bytes)
    s3.put_object(Bucket=BUCKET, Key=key.replace(".bin", ".json"), Body=str(meta).encode())
    return {"s3": f"s3://{BUCKET}/{key}"}
