#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

cd /opt/translate
JOB="${1:?Pass the exact completed or checkpoint job ID}"
[[ "$JOB" =~ ^[a-f0-9]{12}$ ]] || { echo 'Pass the exact 12-character job ID.'; exit 2; }
RESULT=0
if bash scripts/audit_tarjomeh_v2216_reports.sh "$JOB"; then
    echo "Memory/style/QA collector completed."
else
    RESULT=$?
    echo "Memory/style/QA collector failed with exit code $RESULT; running companion too."
fi
if bash scripts/audit_tarjomeh_v2216_companion.sh "$JOB"; then
    echo "Independent companion collector completed."
else
    RESULT=$?
    echo "Independent companion collector failed with exit code $RESULT."
fi
echo "Audit exit code: $RESULT. REVIEW is not semantic certification."
echo "Include the full v22.16 deploy log and the downloaded DOCX with both audit sets."
exit "$RESULT"
