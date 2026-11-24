"""Pull dec1 results from every pod in the registries into runs/dec1/<pod>/ and print a status line per pod.
Usage: python pull_dec1.py [--no-pull]"""
import json, os, re, subprocess, sys
REG = [os.path.expanduser(p) for p in ("~/Desktop/master-thesis-topic/pods-dec1-b.json", "~/Desktop/master-thesis-topic/pods-dec1-ac.json")]
pods = []
for r in REG:
    if os.path.exists(r):
        pods += json.load(open(r))["pods"]
pull = "--no-pull" not in sys.argv
os.chdir(os.path.expanduser("~/Desktop/fu-ml/coin"))
for p in pods:
    ssh = p.get("ssh", "")
    if not ssh.startswith("ssh "):
        print(f"{p['name']}: no ssh yet"); continue
    cmd = ssh.split() + ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15"]
    dest = f"runs/dec1/{p['name']}"; os.makedirs(dest, exist_ok=True)
    if pull:
        tar = subprocess.run(cmd + ["cd /workspace/coin && tar cz $(ls runs/d1_* 2>/dev/null) logs 2>/dev/null"], capture_output=True)
        if tar.returncode == 0 and tar.stdout:
            subprocess.run(["tar", "xz", "-C", dest], input=tar.stdout, capture_output=True)
    st = subprocess.run(cmd + ["cd /workspace/coin && ls runs/d1_*.json 2>/dev/null | wc -l; for f in logs/launch_*.log logs/queue.log; do [ -f $f ] && echo \"$f: $(tail -1 $f)\"; done; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader"], capture_output=True, text=True)
    out = " | ".join(l.strip() for l in st.stdout.splitlines() if l.strip()) or st.stderr.strip()[:200]
    local = len([f for f in os.listdir(f"{dest}/runs") if f.endswith(".json")]) if os.path.isdir(f"{dest}/runs") else 0
    print(f"{p['name']} [{p.get('card','?')}]: pulled {local} json | pod: {out}")
