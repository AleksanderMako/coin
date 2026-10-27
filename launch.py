"""Run a list of shell commands N at a time; skip any whose --tag output already exists.
Usage: python launch.py jobs.txt --parallel 6 [--out runs] [--logs logs]
Each line of jobs.txt is one command containing '--tag NAME'. Blank lines and '#' lines are ignored.
"""
import argparse
import os
import re
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("jobs")
p.add_argument("--parallel", type=int, default=1)
p.add_argument("--out", default="runs")
p.add_argument("--logs", default="logs")
args = p.parse_args()

os.makedirs(args.logs, exist_ok=True)
lines = [l.strip() for l in open(args.jobs) if l.strip() and not l.startswith("#")]
queue = []
for cmd in lines:
    m = re.search(r"--tag\s+(\S+)", cmd)
    tag = m.group(1) if m else None
    done = tag and os.path.exists(os.path.join(args.out, tag + ".json"))
    if done:
        print(f"skip (done): {tag}")
    else:
        queue.append((tag or f"job{len(queue)}", cmd))
print(f"{len(queue)} jobs to run, {args.parallel} at a time")

running = []
t0 = time.time()
while queue or running:
    while queue and len(running) < args.parallel:
        tag, cmd = queue.pop(0)
        log = open(os.path.join(args.logs, tag + ".log"), "w")
        proc = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
        running.append((tag, proc, log, time.time()))
        print(f"[{time.time() - t0:7.0f}s] start {tag}", flush=True)
    time.sleep(5)
    still = []
    for tag, proc, log, started in running:
        if proc.poll() is None:
            still.append((tag, proc, log, started))
        else:
            log.close()
            print(f"[{time.time() - t0:7.0f}s] {'done' if proc.returncode == 0 else f'FAILED({proc.returncode})'} "
                  f"{tag} after {time.time() - started:.0f}s", flush=True)
    running = still
print(f"all done in {time.time() - t0:.0f}s")
