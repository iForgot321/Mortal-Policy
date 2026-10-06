# tools/fleet: rented GPU boxes on Vast.ai

How to rent, use and dispose of GPU boxes with the `vastai` CLI, and the scripts that run on them.
Measured results and host choices are in `experiments/fleet-host-benchmarks.md`; the plan is `03`.

**Rules**
- **No credentials on a box** (no GitHub, Claude or cloud keys): a community host has root on it.
- **No Tenhou data on community boxes** (D23). Benchmarks and RL self-play need none.
- **No published ports.** SSH only; tensorboard through an SSH tunnel. `server.py` has no auth.
- **Commit before syncing:** the box gets the working tree as is, and results record the commit.
- **Destroy a box as soon as its results are copied back.** An idle box keeps billing (day one: 7 idle
  hours cost $3.40, against $0.40 of work).

| File | What it does |
|---|---|
| `setup_box.sh` | Runs on the box: packages, Python/torch/GPU checks, Rust, pinned packages, libriichi build, frozen-input hashes, box configs, tests |
| `box_bench.sh` | Runs here: drives one box end to end (wait, sync, setup, benchmarks, copy back, destroy) |
| `bench_1v3.py` | Eval-mode hanchan/h; `--shadow` / `--compile` price other network sizes |
| `bench_matrix.sh` | Multi-process actors and the reduced inference matrix (one box) |
| `bench_featurize.py` | Featurization rate from Tenhou logs (workstation or secure-tier box only) |
| `requirements-box.txt`, `frozen.sha256` | Pinned box packages; hashes of the frozen checkpoints |

## One-time setup

```bash
pip install --user vastai                 # 1.8.3 here
vastai set api-key <key>                  # from the console (Account → Keys); stored in ~/.config/vastai/vast_api_key
vastai show user --raw | grep credit      # prepaid credit, in $

# a dedicated, passphrase-less key: the shell that drives the boxes has no ssh-agent
ssh-keygen -t ed25519 -f ~/.ssh/vast_ed25519 -N "" -C vast-fleet
vastai create ssh-key "$(cat ~/.ssh/vast_ed25519.pub)"
vastai show ssh-keys
```

Account keys are copied into an instance **when it is created**. For a key added later:
`vastai attach ssh <instance_id> "$(cat ~/.ssh/vast_ed25519.pub)"`.

## Finding an offer

```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 cpu_cores_effective>=32 cpu_ram>=64 cuda_vers>=12.4
    reliability>0.98 disk_space>=50 direct_port_count>=1 inet_down>=500' --storage 50 -o dph_total
```

| Field | Meaning |
|---|---|
| `gpu_name=RTX_4090` | Spaces become underscores |
| `cpu_cores_effective` | The vCPUs you get (the box quota; ~30.7 for "32") |
| `cpu_ram` | GB of system RAM |
| `cuda_vers` | Highest CUDA the host driver supports; the image needs ≥ 12.4 |
| `reliability` | Host uptime score |
| `direct_port_count` | Needed for `--direct` SSH |
| `inet_down` | Mb/s; the image pull is several GB |
| `geolocation in [US,CA]` | Country filter |
| `datacenter=True` | Vast Secure Cloud only (vetted datacenters; the tier for Tenhou data, D23) |

- The default query adds `external=false rentable=true verified=true`; `-n` drops it.
- `--storage 50` makes `dph_total` include 50 GB of disk; `--raw` prints JSON (fields such as
  `machine_id`, `cpu_name`, `dph_total`, `internet_down_cost_per_tb`).
- **What to rent:** RTX 4090 only (torch 2.6 cu124 has no Blackwell kernels, so no 5090s), and
  prefer Zen 4 CPUs (EPYC 9xx4, Ryzen 7000/9000): the actor speed follows the CPU generation.
- **Offer IDs change** when hosts relist, and `vastai search offers 'id=<offer>'` matches nothing.
  Pick a machine, then fetch its current offer right before creating:
  `vastai search offers 'machine_id=<m> num_gpus=1 gpu_name=RTX_4090' --storage 50 --raw`.

## Creating an instance

```bash
vastai create instance <offer_id> \
  --image vastai/pytorch:2.6.0-cuda-12.4.1-py312-22.04 \
  --env '-e OPEN_BUTTON_PORT="1111" -e OPEN_BUTTON_TOKEN="1" -e JUPYTER_DIR="/" -e DATA_DIRECTORY="/workspace/" -e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard"' \
  --onstart-cmd 'entrypoint.sh' --disk 50 --ssh --direct --label mp-<purpose> --raw
```

- **Billing starts here.** The reply's `new_contract` is the instance ID. (It also holds an
  `instance_api_key`; don't log or share it.)
- This is the Vast PyTorch template **minus every `-p` port mapping**: with `--direct` each `-p`
  publishes that service (Jupyter, Syncthing, the portal) on the host's public IP. `--ssh` maps SSH.
- The image tag pins torch 2.6.0 + CUDA 12.4 + Python 3.12, matching the workstation. Never use an
  `-auto` tag.

## Watching and connecting

```bash
vastai show instances                     # table of your instances
vastai show instance <id> --raw           # JSON: actual_status, status_msg, public_ipaddr, ports
vastai ssh-url <id>                       # ssh://root@<ip>:<port>
vastai logs <id>                          # container logs
```

- `actual_status` goes `loading` (image pull; `status_msg` shows progress) → `running`, usually in
  2–5 minutes. Two boxes never left `loading` in 25 minutes; destroy and pick another host.
- SSH: `ports["22/tcp"][0]["HostPort"]` on `public_ipaddr`:

  ```bash
  ssh -i ~/.ssh/vast_ed25519 -p <port> -o UserKnownHostsFile=~/.ssh/known_hosts_vast_<id> root@<ip>
  ```

  Use a known-hosts file per instance: hosts reuse IP:port pairs across instances, so a shared file
  raises "host key changed".
- Inside, the image's Python is `/venv/main/bin/python`; it is not on `PATH` in a non-interactive
  ssh session. Long jobs go in `tmux`; tensorboard via `ssh -L 6006:localhost:6006 …`.

## Running work

**Automated** (preferred): one detached driver per instance, which always destroys its box.

```bash
setsid -f tools/fleet/box_bench.sh <id> 4090-<cpu> none > /tmp/drv-<id>.out 2>&1 < /dev/null
pgrep -o -f '[b]ox_bench.sh <id>'         # the PID to wait on
```

Modes: `none` (setup + 15-min eval benchmark), `full` (+ py-spy profile, P0 at 800- and 2,000-game
batches), `matrix` (setup + `bench_matrix.sh`; launch with `HARD_LIMIT=10800`). Results land in
`workdir/fleet/<date>-vast-<label>/`; the driver's log is `driver.log` there. Its watchdog destroys
the box after `HARD_LIMIT` seconds (default 7200) no matter what.

**By hand:**

```bash
SSHC="ssh -i ~/.ssh/vast_ed25519 -p <port> -o UserKnownHostsFile=~/.ssh/known_hosts_vast_<id>"
rsync -az -e "$SSHC" --filter=':- .gitignore' --exclude=.claude/ --exclude=.pytest_cache/ ./ root@<ip>:/workspace/Mortal-Policy/
rsync -azR -e "$SSHC" workdir/pilot/qs-a/pilot-qs-a-final.pth workdir/grp-v2/grp-v2-frozen.pth root@<ip>:/workspace/Mortal-Policy/
$SSHC root@<ip> "cd /workspace/Mortal-Policy && tmux new -d -s setup 'tools/fleet/setup_box.sh > workdir/box/setup.log 2>&1'"
# … work …
rsync -az -e "$SSHC" root@<ip>:/workspace/Mortal-Policy/workdir/box/ workdir/fleet/<run>/box/
```

## Stopping and destroying

```bash
vastai destroy instance <id> -y           # irreversible; without -y it prompts, and a script aborts
vastai stop instance <id>                 # GPU billing stops, disk keeps billing; restarting needs that GPU free
vastai show instances                     # confirm nothing is left
```

## Costs

- `dph_total` covers GPU, CPU share and the disk size passed to `--storage`.
- Bandwidth is priced per host, per TB (`internet_down_cost_per_tb`, `internet_up_cost_per_tb`;
  typically ~$3–10/TB, some hosts $27–55/TB). It matters for actors uploading games, not for benchmarks.
- Disk keeps billing while an instance is stopped; only `destroy` ends all charges.
- A benchmark box has cost $0.40–0.70 per run; the running total is in `STATUS.md`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Permission denied (publickey)` | Key not on the account when the instance was created: `vastai attach ssh <id> "<pubkey>"` |
| `python: command not found` | Use `/venv/main/bin/python` |
| `git describe` says `-dirty` on the box | CRLF docs in the workstation tree; `setup_box.sh` sets `core.autocrlf input` |
| `setup_box.sh` stops at the GPU check | A Blackwell GPU (5090): torch 2.6 cu124 can't run it |
| Stuck in `loading` | Slow or stalled image pull on that host; destroy, pick another |
| A waiter "finishes" at once while the job runs | It watched the wrong PID; see the `box_bench.sh` launch lines above |
