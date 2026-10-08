#!/usr/bin/env python3
"""Run a Python script under a shared, machine-wide memory budget (default 20 GB total).

Usage: run_budget.py --est-gb X [--timeout S] [--budget-gb 20] script.py [args...]

Each job reserves --est-gb from a shared ledger before starting. Jobs run concurrently
as long as the sum of reservations stays <= --budget-gb; otherwise they wait. A job whose
resident memory (including children) exceeds its own reservation is killed, so the total
in-use memory of all budgeted jobs can never exceed the budget. Runs with cwd = <repo>/batch
and the project venv.
"""
import fcntl, json, os, signal, subprocess, sys, tempfile, time

# The ledger lives in the system temp dir (shared by every job on this machine, never in the repo);
# the repo root is three levels above this file (.claude/skills/local-tests/).
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LEDGER = os.path.join(tempfile.gettempdir(), "coron_mem_ledger.json")
LOCK = LEDGER + ".lock"
PY = os.path.join(REPO, ".venv", "bin", "python")
CWD = os.path.join(REPO, "batch")

args = sys.argv[1:]
est_gb, timeout, budget_gb = None, 1800.0, 20.0
while args and args[0].startswith("--"):
    flag, val, args = args[0], args[1], args[2:]
    if flag == "--est-gb":
        est_gb = float(val)
    elif flag == "--timeout":
        timeout = float(val)
    elif flag == "--budget-gb":
        budget_gb = min(float(val), 20.0)
if est_gb is None or not args:
    sys.exit(__doc__)
if est_gb > budget_gb:
    sys.exit(f"--est-gb {est_gb} exceeds the total budget {budget_gb}")

env = dict(os.environ,
           XLA_PYTHON_CLIENT_PREALLOCATE="false",
           XLA_PYTHON_CLIENT_ALLOCATOR="platform",
           OMP_NUM_THREADS="3",
           XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=3")


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def ledger_update(fn):
    with open(LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            entries = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {}
        except json.JSONDecodeError:
            entries = {}
        entries = {p: g for p, g in entries.items() if alive(int(p))}
        result = fn(entries)
        json.dump(entries, open(LEDGER, "w"))
        return result


def try_reserve(entries):
    used = sum(entries.values())
    if used + est_gb <= budget_gb:
        entries[str(os.getpid())] = est_gb
        return True, used
    return False, used


def tree_rss_kb(pid):
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    rows = [tuple(map(int, l.split())) for l in out.splitlines() if l.strip()]
    kids, rss = {}, {}
    for p, pp, r in rows:
        kids.setdefault(pp, []).append(p)
        rss[p] = r
    total, stack = 0, [pid]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack.extend(kids.get(p, []))
    return total


t_wait = time.time()
while True:
    ok, used = ledger_update(try_reserve)
    if ok:
        break
    time.sleep(2)
print(f"[run_budget] reserved {est_gb} GB (others using {used:.1f}/{budget_gb} GB) after {time.time()-t_wait:.0f}s wait", flush=True)

try:
    proc = subprocess.Popen([PY, *args], cwd=CWD, env=env, start_new_session=True)
    t0, peak, reason = time.time(), 0, None
    while proc.poll() is None:
        kb = tree_rss_kb(proc.pid)
        peak = max(peak, kb)
        if kb > est_gb * 1024 ** 2:
            reason = f"RSS {kb/1024**2:.2f} GB > reservation {est_gb} GB (re-run with a larger --est-gb or a smaller problem)"
        elif time.time() - t0 > timeout:
            reason = f"timeout {timeout}s"
        if reason:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            break
        time.sleep(0.25)
    print(f"[run_budget] exit={proc.returncode} peak_rss={peak/1024**2:.2f} GB wall={time.time()-t0:.0f}s"
          + (f" KILLED: {reason}" if reason else ""), flush=True)
finally:
    ledger_update(lambda e: e.pop(str(os.getpid()), None))
sys.exit(137 if reason else proc.returncode)
