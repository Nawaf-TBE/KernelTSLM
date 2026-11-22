#!/bin/bash
# From your machine: copy every run's results from the GPU VM to runs/. Left out: the models (13.7 GB each)
# and the inputs that upload.sh sent (runs/baselines, runs/temporal_qa, runs/usecases), so the local copies stay.
#   gpu/pull.sh <user>@<vm-ip>          # once
#   gpu/pull.sh <user>@<vm-ip> --loop   # every 5 minutes until multitask.sh is done
HOST=${1:?usage: pull.sh <user>@<vm-ip> [--loop]}
MARKER=multitask.done
REPO=$(cd "$(dirname "$0")/.." && pwd)
pull() {
  # NOTE: GNU tar on the VM: a pattern without a slash matches that name at any depth
  # NOTE: the queue's own files (multitask.log, .done, .pid, .current) are left out, so that several VMs do not
  # overwrite each other locally; the log goes to runs/logs/<vm-ip>.log instead
  ssh -o ConnectTimeout=15 "$HOST" \
    'cd hackathon/runs && tar czf - --exclude=model --exclude=baselines --exclude=temporal_qa --exclude=usecases --exclude="multitask.*" *' \
    | tar xzf - -C "$REPO/runs" && echo "$(date +%H:%M) copied"
  mkdir -p "$REPO/runs/logs"
  ssh -o ConnectTimeout=15 "$HOST" 'cat hackathon/runs/multitask.log 2>/dev/null' > "$REPO/runs/logs/${HOST#*@}.log"
}
pull
if [ "$2" = "--loop" ]; then
  # NOTE: the marker is checked on the VM, so the done marker of an earlier queue here does not stop the loop
  until ssh -o ConnectTimeout=15 "$HOST" "test -f hackathon/runs/$MARKER"; do sleep 300; pull; done
  pull
  echo "queue done ($MARKER); all results are in $REPO/runs"
fi
