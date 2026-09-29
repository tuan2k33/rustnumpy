"""Aggregate the JSONL/stat files written by rnp_shim into a readable report."""

import collections
import glob
import json
import os
import sys

log_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rnp_shim_log"
stats = collections.defaultdict(lambda: collections.Counter())
for path in glob.glob(os.path.join(log_dir, "stats-*.json")):
    for name, outcome, n in json.load(open(path)):
        stats[name][outcome] += n

issues = collections.defaultdict(list)
for path in glob.glob(os.path.join(log_dir, "*.jsonl")):
    for line in open(path):
        rec = json.loads(line)
        issues[(rec["func"], rec["outcome"])].append(rec)

print(f"{'function':28} {'match':>8} {'mismatch':>9} {'we-err':>7} {'np-err':>7} {'fallback':>9} {'skipped':>8}")
tot = collections.Counter()
for name in sorted(stats):
    c = stats[name]
    print(f"{name:28} {c['match']:8} {c['mismatch']:9} {c['we_error_numpy_ok']:7} {c['numpy_error_we_ok']:7} {c['fallback']:9} {c['skipped']:8}")
    tot.update(c)
print(f"{'TOTAL':28} {tot['match']:8} {tot['mismatch']:9} {tot['we_error_numpy_ok']:7} {tot['numpy_error_we_ok']:7} {tot['fallback']:9} {tot['skipped']:8}")
print(f"\nboth-error (same failure as NumPy): {tot['both_error']}   errstate/warning-only (NumPy raises FP errors, Rust has no errstate): {tot['errstate_or_warning']}")

print("\n=== divergences (grouped by function / kind / reason) ===")
for (func, outcome), recs in sorted(issues.items()):
    by_reason = collections.Counter(r.get("detail", "") for r in recs)
    print(f"\n{func} [{outcome}] x{len(recs)}")
    for reason, n in by_reason.most_common(6):
        example = next(r for r in recs if r.get("detail", "") == reason)
        print(f"   {n:5}  {reason[:110]}")
        print(f"          e.g. {', '.join(example['args'])[:150]}")
        if "ours" in example:
            print(f"          inputs={example['inputs']}\n          ours   ={example['ours']}\n          numpy  ={example['theirs']}")
