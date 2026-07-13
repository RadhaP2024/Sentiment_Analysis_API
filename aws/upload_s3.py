import boto3

s3 = boto3.client("s3")

bucket_name = "your-bucket-name"

s3.upload_file(
    "model/model.safetensors",
    bucket_name,
    "model/model.safetensors"
)

print("Upload Successful")