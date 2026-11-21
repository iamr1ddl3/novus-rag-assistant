#!/usr/bin/env bash
# Tear down the Novus RAG assistant stack (acmera-kb-<env>).
# RDS has DeletionPolicy: Snapshot, so a final snapshot is kept; the next
# up.sh starts with an empty database and re-ingests the corpus.
# ECR images, the app-keys secret and earlier snapshots are not touched.
#
# The Support Agent calls this API, so take it down first
# (novus-support-agent/deploy/down.sh). FORCE=1 skips that check.
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
STACK="acmera-kb-${ENVIRONMENT}"
SUPPORT_STACK="acmera-support-${ENVIRONMENT}"

if [[ "${FORCE:-0}" != "1" ]] && aws cloudformation describe-stacks --region "$REGION" --stack-name "$SUPPORT_STACK" >/dev/null 2>&1; then
  echo "${SUPPORT_STACK} is still up and depends on this API. Run its down.sh first, or set FORCE=1." >&2
  exit 1
fi

echo "Deleting ${STACK} in ${REGION} (takes about 10 minutes) ..."
aws cloudformation delete-stack --region "$REGION" --stack-name "$STACK"
aws cloudformation wait stack-delete-complete --region "$REGION" --stack-name "$STACK"
echo "${STACK} deleted. Billing for it has stopped (except image and snapshot storage)."
