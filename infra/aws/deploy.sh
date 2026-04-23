#!/bin/bash
# deploy.sh — build, push to ECR, and create/update Lambda + API Gateway
#
# Usage:
#   bash infra/aws/deploy.sh <account_id> <region>
#
# Example:
#   bash infra/aws/deploy.sh 123456789012 eu-south-1
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker running
#   - HF_TOKEN set in environment or .env file at project root

set -euo pipefail  # exit on error, undefined vars, pipe failures

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ACCOUNT_ID=${1:?"Usage: $0 <account_id> <region>"}
REGION=${2:?"Usage: $0 <account_id> <region>"}
FUNCTION_NAME="agentic-vision-orchestrator"
REPO_NAME="agentic-vision-cloud-orchestrator"
ROLE_NAME="lambda-execution-role"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest"
MEMORY_MB=1024
TIMEOUT_S=60

# ---------------------------------------------------------------------------
# 1. Resolve HF_TOKEN (env var takes priority over .env file)
# ---------------------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
    ENV_FILE="$(dirname "$0")/../../.env"
    if [ -f "$ENV_FILE" ]; then
        HF_TOKEN=$(grep -E "^HF_TOKEN=" "$ENV_FILE" | cut -d= -f2- | tr -d '[:space:]')
    fi
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Export it or add it to .env." >&2
    exit 1
fi

echo ">>> [1/6] HF_TOKEN resolved."

# ---------------------------------------------------------------------------
# 2. Create ECR repository (idempotent — silently skips if already exists)
# ---------------------------------------------------------------------------
echo ">>> [2/6] Ensuring ECR repository exists..."
aws ecr describe-repositories \
    --repository-names "$REPO_NAME" \
    --region "$REGION" > /dev/null 2>&1 \
|| aws ecr create-repository \
    --repository-name "$REPO_NAME" \
    --region "$REGION" > /dev/null

# ---------------------------------------------------------------------------
# 3. Build, tag, and push Docker image to ECR
# ---------------------------------------------------------------------------
echo ">>> [3/6] Building Docker image..."
docker build -t "$REPO_NAME" -f "$(dirname "$0")/../../Dockerfile" \
    "$(dirname "$0")/../.."

echo ">>> Logging in to ECR..."
aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin \
      "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo ">>> Tagging and pushing image..."
docker tag "${REPO_NAME}:latest" "$IMAGE_URI"
docker push "$IMAGE_URI"

# Verify the image is actually present in ECR before proceeding
echo ">>> Verifying image in ECR..."
aws ecr describe-images \
    --repository-name "$REPO_NAME" \
    --image-ids imageTag=latest \
    --region "$REGION" > /dev/null \
|| { echo "ERROR: Image not found in ECR after push. Aborting." >&2; exit 1; }

echo ">>> [3/6] Image pushed and verified."

# ---------------------------------------------------------------------------
# 4. Create or update IAM role (idempotent)
# ---------------------------------------------------------------------------
echo ">>> [4/6] Ensuring IAM role exists..."
ROLE_ARN=$(aws iam get-role \
    --role-name "$ROLE_NAME" \
    --query "Role.Arn" \
    --output text 2>/dev/null) \
|| {
    echo ">>> Creating IAM role..."
    ROLE_ARN=$(aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document '{
            "Version":"2012-10-17",
            "Statement":[{
                "Effect":"Allow",
                "Principal":{"Service":"lambda.amazonaws.com"},
                "Action":"sts:AssumeRole"
            }]
        }' \
        --query "Role.Arn" \
        --output text)

    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

    # Wait for IAM role to propagate using AWS waiter (reliable, no arbitrary sleep)
    echo ">>> Waiting for IAM role to propagate..."
    aws iam wait role-exists --role-name "$ROLE_NAME"
    sleep 5  # minimal extra wait for policy attachment propagation
}

echo ">>> IAM role ARN: $ROLE_ARN"

# ---------------------------------------------------------------------------
# 5. Create or update Lambda function (idempotent)
# ---------------------------------------------------------------------------
echo ">>> [5/6] Deploying Lambda function..."

if aws lambda get-function \
    --function-name "$FUNCTION_NAME" \
    --region "$REGION" > /dev/null 2>&1; then

    echo ">>> Function exists — updating image and config..."
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --image-uri "$IMAGE_URI" \
        --region "$REGION" > /dev/null

    # Wait for update to complete before updating config
    aws lambda wait function-updated \
        --function-name "$FUNCTION_NAME" \
        --region "$REGION"

    aws lambda update-function-configuration \
        --function-name "$FUNCTION_NAME" \
        --timeout "$TIMEOUT_S" \
        --memory-size "$MEMORY_MB" \
        --environment "Variables={HF_TOKEN=${HF_TOKEN}}" \
        --region "$REGION" > /dev/null

    aws lambda wait function-updated \
        --function-name "$FUNCTION_NAME" \
        --region "$REGION"

else
    echo ">>> Function does not exist — creating..."
    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --package-type Image \
        --code "ImageUri=${IMAGE_URI}" \
        --role "$ROLE_ARN" \
        --timeout "$TIMEOUT_S" \
        --memory-size "$MEMORY_MB" \
        --environment "Variables={HF_TOKEN=${HF_TOKEN}}" \
        --region "$REGION" > /dev/null

    aws lambda wait function-active \
        --function-name "$FUNCTION_NAME" \
        --region "$REGION"
fi

echo ">>> Lambda deployed."

# ---------------------------------------------------------------------------
# 6. Create or update API Gateway (idempotent)
# ---------------------------------------------------------------------------
echo ">>> [6/6] Ensuring API Gateway exists..."
LAMBDA_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"

# Check if an API with this name already exists
API_ID=$(aws apigatewayv2 get-apis \
    --region "$REGION" \
    --query "Items[?Name=='${FUNCTION_NAME}-api'].ApiId" \
    --output text 2>/dev/null)

if [ -z "$API_ID" ] || [ "$API_ID" == "None" ]; then
    echo ">>> Creating API Gateway..."
    API_ID=$(aws apigatewayv2 create-api \
        --name "${FUNCTION_NAME}-api" \
        --protocol-type HTTP \
        --region "$REGION" \
        --query "ApiId" \
        --output text)

    INTEGRATION_ID=$(aws apigatewayv2 create-integration \
        --api-id "$API_ID" \
        --integration-type AWS_PROXY \
        --integration-uri "$LAMBDA_ARN" \
        --payload-format-version "2.0" \
        --region "$REGION" \
        --query "IntegrationId" \
        --output text)

    aws apigatewayv2 create-route \
        --api-id "$API_ID" \
        --route-key "ANY /{proxy+}" \
        --target "integrations/${INTEGRATION_ID}" \
        --region "$REGION" > /dev/null

    aws apigatewayv2 create-stage \
        --api-id "$API_ID" \
        --stage-name prod \
        --auto-deploy \
        --region "$REGION" > /dev/null

    # Grant API Gateway permission to invoke Lambda
    aws lambda add-permission \
        --function-name "$FUNCTION_NAME" \
        --statement-id "apigateway-invoke-${API_ID}" \
        --action lambda:InvokeFunction \
        --principal apigateway.amazonaws.com \
        --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*" \
        --region "$REGION" > /dev/null

    echo ">>> API Gateway created."
else
    echo ">>> API Gateway already exists (ID: ${API_ID}). No changes needed."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
ENDPOINT="https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod"

echo ""
echo "============================================================"
echo "  Deploy complete!"
echo "  Endpoint : ${ENDPOINT}"
echo "  Health   : ${ENDPOINT}/health"
echo "  Predict  : ${ENDPOINT}/predict"
echo "============================================================"
echo ""
echo "Quick test:"
echo "  curl ${ENDPOINT}/health"
