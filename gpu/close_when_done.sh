#!/bin/bash
# From the Mac: wait for a VM's queue to finish, copy its results, check them, then delete the VM (and its managed
# boot disk) with the Nebius CLI. Deletes nothing if a check fails.
#   gpu/close_when_done.sh <user>@<ip> <instance-id> <job> [<job> ...]
HOST=$1; ID=$2; shift 2; JOBS=("$@")
export PATH="$HOME/.nebius/bin:$PATH"
log() { echo "$(date -u +%H:%M) [$HOST] $*"; }
until ssh -o ConnectTimeout=15 -o BatchMode=yes "$HOST" 'test -f hackathon/runs/multitask.done' 2>/dev/null; do sleep 60; done
log "queue done"
~/hackathon/gpu/pull.sh "$HOST" > /dev/null 2>&1 || { log "KEEP: the copy failed"; exit 1; }
for job in "${JOBS[@]}"; do
  m=~/hackathon/runs/$job/metrics.json
  if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$m')).get('complete') else 1)" 2>/dev/null; then
    log "KEEP: $job is missing or incomplete on the Mac"; exit 1
  fi
done
log "results copied and complete: ${JOBS[*]}"
nebius compute instance delete --id "$ID" > /dev/null 2>&1 || { log "KEEP: the delete command failed"; exit 1; }
log "deleted instance $ID"
