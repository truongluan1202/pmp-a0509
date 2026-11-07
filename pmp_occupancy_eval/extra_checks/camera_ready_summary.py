"""Pool camera_ready_variants*.json into the Table 3 layout."""
import json, sys, numpy as np, collections
rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'extra_checks/camera_ready_variants.json'))
by = collections.defaultdict(list)
for r in rows: by[r['policy']].append(r)
print(f"{'policy':<14}{'rec viol':>9}{'viol/1e6 range':>18}{'anchor':>8}{'FSE static %':>13}{'FSE gap %':>10}{'Occ mean %':>11}  per-recording occ")
for p, rs in by.items():
    v = [1e6 * r['violations'] / ((r['frames'] - 1) * r['points']) for r in rs if r['rep'] == 0]
    va = sum(r['violations_anchor'] for r in rs)
    st = [r for r in rs if 'static_ref' in r]
    fs = 100 * sum(r['static_missed'] for r in st) / sum(r['static_ref'] for r in st) if st else float('nan')
    fg = 100 * sum(r['gap_missed'] for r in rs) / sum(r['gap_ref'] for r in rs)
    occ = 100 * np.mean([r['occ'] for r in rs])
    per = {r['bag']: round(100 * r['occ']) for r in rs if r['rep'] == 0}
    print(f"{p:<14}{sum(x > 0 for x in v):>5} of {len(v)}{min(v):>8.1f} to {max(v):<7.0f}{va:>8}{fs:>13.2f}{fg:>10.3f}{occ:>11.1f}  {per}")
