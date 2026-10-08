#!/usr/bin/env bash
# Hold an actor's tunnel to the learner's server.py (D48): 127.0.0.1:$LOCAL_PORT here is forwarded
# to 127.0.0.1:$REMOTE_PORT on the learner, over the learner's SSH port, as its restricted tunnel
# user. client.py's backoff covers the seconds between a drop and the reconnect.
#
# The target is re-read from the env file before every attempt, so `fleet.py join` can repoint a
# running tunnel at a replacement learner: it rewrites the file and kills the PID in <name>.pid.
# Run by fleet.py in the actor's `tunnel` tmux session.
#
# Usage: box_tunnel.sh <tunnel.env>
#   tunnel.env sets LEARNER_IP, LEARNER_SSH_PORT, TUNNEL_USER, LOCAL_PORT, REMOTE_PORT, KEY and
#   KNOWN_HOSTS (holding the learner's pinned ed25519 host key)
set -u
ENV_FILE=${1:?usage: box_tunnel.sh <tunnel.env>}
PIDFILE=${ENV_FILE%.env}.pid
while true; do
    # shellcheck source=/dev/null
    if [[ -r $ENV_FILE ]] && source "$ENV_FILE"; then
        echo "$(date -Is) connecting to $TUNNEL_USER@$LEARNER_IP:$LEARNER_SSH_PORT"
        ssh -N -i "$KEY" -p "$LEARNER_SSH_PORT" -o IdentitiesOnly=yes -o BatchMode=yes \
            -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$KNOWN_HOSTS" \
            -o HostKeyAlgorithms=ssh-ed25519 -o ExitOnForwardFailure=yes \
            -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ConnectTimeout=20 \
            -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" "$TUNNEL_USER@$LEARNER_IP" &
        echo $! > "$PIDFILE"
        wait $!
        echo "$(date -Is) ssh exited $?"
    else
        echo "$(date -Is) can't read $ENV_FILE"
    fi
    sleep 5
done
