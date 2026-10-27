#!/bin/bash
# Wait for the running launcher to finish, then run the given job lists one after another.
# Usage: setsid nohup bash queue.sh jobs/a.txt jobs/b.txt > logs/queue.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
while pgrep -f "python [l]aunch.py" >/dev/null; do sleep 30; done
for f in "$@"; do
  echo "== $(date -u +%H:%M) starting $f"
  python launch.py "$f" --parallel 1
done
echo "== $(date -u +%H:%M) queue done"
