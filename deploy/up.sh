#!/usr/bin/env bash
# Bring up the Novus RAG assistant stack (acmera-kb-<env>) and load the corpus.
#
#   ./deploy/up.sh            deploy with the image already in ECR
#   ./deploy/up.sh --build    rebuild + push the image first (needs Docker)
#
# Env overrides: AWS_REGION (ap-south-1), ENVIRONMENT (dev),
#   APP_KEYS_SECRET (novus/dev/app-keys), USE_SPOT (true; "false" for interviews)
# Prerequisite: a Secrets Manager secret APP_KEYS_SECRET holding JSON keys
#   OPENAI_API_KEY, COHERE_API_KEY (may be ""), LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
SECRET_ID="${APP_KEYS_SECRET:-novus/dev/app-keys}"
USE_SPOT="${USE_SPOT:-true}"
STACK="acmera-kb-${ENVIRONMENT}"
ECR_REPO="novus-rag-assistant"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
echo "Account ${ACCOUNT}, region ${REGION}, stack ${STACK}, spot=${USE_SPOT}"

if [[ "${1:-}" == "--build" ]]; then
  REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
  aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
  docker build --platform linux/amd64 -t "${REGISTRY}/${ECR_REPO}:latest" "$ROOT"
  docker push "${REGISTRY}/${ECR_REPO}:latest"
fi

SECRET_ARN="$(aws secretsmanager describe-secret --region "$REGION" --secret-id "$SECRET_ID" --query ARN --output text)"

aws cloudformation deploy \
  --region "$REGION" \
  --stack-name "$STACK" \
  --template-file "${ROOT}/template.yaml" \
  --capabilities CAPABILITY_NAMED_IAM \
  --no-fail-on-empty-changeset \
  --parameter-overrides "Environment=${ENVIRONMENT}" "AppKeysSecretArn=${SECRET_ARN}" "UseSpot=${USE_SPOT}"

API="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='APIEndpoint'].OutputValue" --output text)"

echo "Waiting for ${API}/health ..."
for _ in $(seq 1 30); do
  if curl -fsS -m 10 "${API}/health" >/dev/null 2>&1; then break; fi
  sleep 10
done
curl -fsS -m 10 "${API}/health" >/dev/null || { echo "Health check failed; see ECS service events." >&2; exit 1; }

echo "Loading the corpus (setup_db + ingest; idempotent) ..."
curl -fsS -m 600 -X POST "${API}/ingest"; echo
echo "RAG assistant is up: ${API}"
