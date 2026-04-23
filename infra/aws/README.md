# AWS Infrastructure — CV Inference Service

The CV service (ResNet50 breed classifier) is deployed on AWS Lambda
using a Docker container image stored in Amazon ECR.

## Architecture

ECR (container image) → Lambda Function → API Gateway HTTP API

- **Function name**: `agentic-vision-orchestrator`
- **Region**: `eu-south-1`
- **Runtime**: Container image (Python 3.11, Lambda base)
- **Memory**: 1024 MB | **Timeout**: 60s
- **Endpoint**: `https://8r6akcsyx5.execute-api.eu-south-1.amazonaws.com/prod`

## Deploying a new version

```bash
# 1. Build image
docker build -t vision-inference-service -f Dockerfile .

# 2. Login to ECR
aws ecr get-login-password --region eu-south-1 | docker login \
  --username AWS \
  --password-stdin <ACCOUNT_ID>.dkr.ecr.eu-south-1.amazonaws.com

# 3. Tag and push
docker tag vision-inference-service:latest \
  <ACCOUNT_ID>.dkr.ecr.eu-south-1.amazonaws.com/vision-inference-service:latest
docker push \
  <ACCOUNT_ID>.dkr.ecr.eu-south-1.amazonaws.com/vision-inference-service:latest

# 4. Update Lambda to use new image
aws lambda update-function-code \
  --function-name agentic-vision-orchestrator \
  --image-uri <ACCOUNT_ID>.dkr.ecr.eu-south-1.amazonaws.com/vision-inference-service:latest \
  --region eu-south-1
```

## Test endpoints

```bash
curl https://8r6akcsyx5.execute-api.eu-south-1.amazonaws.com/prod/health

curl -X POST https://8r6akcsyx5.execute-api.eu-south-1.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://images.dog.ceo/breeds/beagle/n02088364_10108.jpg"}'
```
