#!/bin/bash
# Final submission run: six bags in parallel, then merged CSVs.
#   bash scripts/run_final.sh
set -e
cd "$(dirname "$0")/.."
test -f config/d_contact_tape.json || { echo "run scripts/tape_to_dcontact.py first"; exit 1; }
mkdir -p final logs
for b in npz_replay_trim/310826_B*.npz; do
  n=$(basename "$b" .npz)
  caffeinate -i python3 -u scripts/run_paper_analysis.py --bags "$b" \
    --config config/eval_params_final.yaml --d-contact config/d_contact_tape.json \
    --out-prefix "final/$n/" > "logs/final_$n.log" 2>&1 &
done
echo "6 runs started; progress: tail -n 2 logs/final_*.log"
wait
python3 - <<'PY'
import csv, glob
for f in ("q1_doseresponse.csv", "q2_ksweep.csv", "gaps.csv"):
    parts = sorted(glob.glob(f"final/310826_*/{f}"))
    rows = [r for p in parts for r in csv.DictReader(open(p))]
    with open(f"final/{f}", "w", newline="") as o:
        w = csv.DictWriter(o, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"final/{f}: {len(parts)} bags, {len(rows)} rows")
PY
caffeinate -i python3 -u scripts/run_mechanism.py --bags npz_replay_trim/310826_B*.npz \
  --config config/eval_params_final.yaml --d-contact config/d_contact_tape.json \
  --out final/mechanism.csv > logs/final_mechanism.log 2>&1
python3 scripts/make_figures.py --run final/ --out final/figs/
echo DONE
