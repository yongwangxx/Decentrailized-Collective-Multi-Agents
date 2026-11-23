"""Running a table of independent queries: parallel execution, re-asking, JSON Lines output, resuming.

Rule for re-asking (single-particle tests, lane runs and repeated queries): a query whose call failed or whose reply
contains no readable answer does not count. Its record is kept in the output with its attempt number, and the same
query is asked again, up to `max_attempts` times; if all attempts fail the run stops. With max_attempts = 1 nothing is
asked again and a reply without a readable answer stays in the data as such.

Output: one JSON object per line and per attempt. Resuming: queries that already have a valid record are skipped.
"""
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def is_valid(r):
    return not r.get("error") and r.get("answer") is not None


def first_valid(records):
    """One record per query id: the first attempt that has no error and a readable answer."""
    out = {}
    for r in records:
        if is_valid(r) and r["id"] not in out:
            out[r["id"]] = r
    return list(out.values())


def run_queries(queries, ask_one, out_path, parallel=8, max_attempts=10, progress_every=50, label=""):
    """ask_one(query, attempt) -> record (a dict that contains at least id, answer and error).
    Appends every attempt to out_path. Returns the number of queries completed in this call."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    done = {r["id"] for r in load(out_path) if is_valid(r) or max_attempts == 1}
    todo = [q for q in queries if q["id"] not in done]
    print(f"{label}{len(todo)} queries to ask ({len(queries) - len(todo)} already in {out_path})", flush=True)
    lock = threading.Lock()
    state = dict(n=0, retries=0, failed=None)

    def work(q):
        for attempt in range(1, max_attempts + 1):
            if state["failed"]:
                return None
            r = ask_one(q, attempt)
            r["attempt"] = attempt
            with lock:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                fo.flush()
                if is_valid(r) or max_attempts == 1:
                    return r
                state["retries"] += 1
        with lock:
            state["failed"] = f"query {q['id']} failed {max_attempts} times in a row (last error: {r.get('error')})"
        return None

    with open(out_path, "a") as fo, ThreadPoolExecutor(parallel) as ex:
        for fu in as_completed([ex.submit(work, q) for q in todo]):
            if fu.result() is None:
                continue
            state["n"] += 1
            if state["n"] % progress_every == 0 or state["n"] == len(todo):
                print(f"  {state['n']}/{len(todo)} done, {state['retries']} asked again", flush=True)
    if state["failed"]:
        print("STOPPED: " + state["failed"], flush=True)
        sys.exit(2)
    return state["n"]


def ask_until_valid(ask_one, max_attempts=10):
    """For the collective runs: call ask_one(attempt) until it returns a valid record.
    Returns (record or None, list of errors of the failed attempts)."""
    errors = []
    for attempt in range(1, max_attempts + 1):
        r = ask_one(attempt)
        if is_valid(r):
            return r, errors
        errors.append(str(r.get("error") or "no readable answer")[:200])
    return None, errors
