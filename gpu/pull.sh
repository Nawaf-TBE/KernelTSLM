#!/bin/bash
# From the Mac: copy every run's results from the GPU VM to ~/hackathon/runs. Left out: the models (13.7 GB each)
# and the inputs that upload.sh sent (runs/baselines, runs/temporal_qa, runs/usecases), so the Mac copies stay.
#   ~/hackathon/gpu/pull.sh <user>@<vm-ip>                               # once
#   caffeinate -i ~/hackathon/gpu/pull.sh <user>@<vm-ip> --loop          # every 5 minutes until multitask.sh is done
#   caffeinate -i ~/hackathon/gpu/pull.sh <user>@<vm-ip> --loop overnight   # the same for overnight.sh (or parallel)
HOST=${1:?usage: pull.sh <user>@<vm-ip> [--loop [multitask|overnight|parallel]]}
MARKER=${3:-multitask}.done
pull() {
  # NOTE: GNU tar on the VM: a pattern without a slash matches that name at any depth
  # NOTE: the queue's own files (multitask.log, .done, .pid, .current) are left out, so that several VMs do not
  # overwrite each other on the Mac; the log goes to runs/logs/<vm-ip>.log instead
  ssh -o ConnectTimeout=15 "$HOST" \
    'cd hackathon/runs && tar czf - --exclude=model --exclude=baselines --exclude=temporal_qa --exclude=usecases --exclude="multitask.*" *' \
    | tar xzf - -C ~/hackathon/runs && echo "$(date +%H:%M) copied"
  mkdir -p ~/hackathon/runs/logs
  ssh -o ConnectTimeout=15 "$HOST" 'cat hackathon/runs/multitask.log 2>/dev/null' > ~/hackathon/runs/logs/"${HOST#*@}".log
}
pull
if [ "$2" = "--loop" ]; then
  # NOTE: the marker is checked on the VM, so the done marker of an earlier queue on the Mac does not stop the loop
  until ssh -o ConnectTimeout=15 "$HOST" "test -f hackathon/runs/$MARKER"; do sleep 300; pull; done
  pull
  echo "queue done ($MARKER); all results are in ~/hackathon/runs"
fi
