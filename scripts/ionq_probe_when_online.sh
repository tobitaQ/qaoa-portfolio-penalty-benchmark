#!/bin/bash
# Poll IonQ Forte-1; when ONLINE, run the probe stage once (user-approved, est. $0, worst case $8.30 cancelled).
cd ~/Work_2026/quantum-finance-research
B=$(aws s3 ls | awk '/amazon-braket-qfr/{print $3}')
for i in $(seq 1 336); do   # up to 7 days at 30-min intervals
  st=$(aws braket get-device --region us-east-1 --device-arn arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1 --query deviceStatus --output text 2>/dev/null)
  echo "$(date -u +%FT%TZ) status=$st"
  if [ "$st" = "ONLINE" ]; then
    .venv/bin/python -W ignore -m experiments.system_paper.es2_observe_cloud --s3-bucket "$B" --yes-bill-me --skip-fallback 2>&1 | sed "s/[0-9]\{12\}/<acct>/g"
    echo PROBE_DONE
    exit 0
  fi
  sleep 1800
done
echo "gave up: device never ONLINE"
