# pmp_risk_eval_v2

Code, derived data and results for the ACRA 2026 paper:

> **A Safety-Aware Perception Layer for Autonomous Patient Repositioning Using a Robotic Arm**
> Dinh Truong Luan Tran, Kanaka Sai Jagarlamudi, Chng Wei Lau, Jacob Desmond, Rhys Tague, Oliver Obst, Anupama Ginige

The package evaluates two occupancy estimators on skeleton tracking data of a
patient handling manikin on a hospital bed. Everything here runs offline with
plain Python. No ROS, camera or robot is needed.

## Names used in the paper and in the code

| Paper            | Code       |
|------------------|------------|
| hold-last        | `hold`     |
| growing-radius   | `proposed` |
| bare             | `B1_neither` |
| sheet            | `B2_contrast_only` |
| pad              | `B4_both`  |
| static reference | `static_ground_truth` |

Each condition has one `static` and one `dynamic` recording, giving six
recordings in total.

## Layout

| Path                 | Contents |
|----------------------|----------|
| `pmp_risk_eval/`     | The library. `occupancy_field.py` holds both estimators, `occlusion_injector.py` the dropout injection, `metrics.py` the scores, `subject_filter.py` the subject selection, `npz_corpus.py` the data loader. |
| `scripts/`           | Entry points. See "Reproducing the results". |
| `tests/`             | Synthetic checks of the estimator properties. |
| `config/`            | `eval_params_final.yaml` is the configuration used for the paper. `tape_measurements.yaml` and `d_contact_tape.json` hold the tape measured manikin dimensions and the contact radii derived from them. |
| `npz_replay_trim/`   | The six recordings as skeleton data (the corpus used in the paper). |
| `final/`             | Results of the final run: merged CSVs, one subfolder per recording, and `figs/`. |
| `sweep_thold/`       | Hold-last timeout sweep (1, 2 and 5 s) with the configuration used for each. |
| `review/`            | Ablations per recording, and static ground truth checks for the static recordings. |
| `extra_checks/`      | Further checks reported in the paper: the no growth row (k = 0), the share of predicted keypoints, and the contact radii against the tape measurements. Scripts and their outputs. |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is a freeze of the environment that produced `final/`.

## Check the installation

```bash
python3 tests/test_occupancy.py     # ends with: all paper v2 checks pass
python3 tests/test_synthetic.py
```

## Reproducing the results

All commands run from the package root.

**Figures only, from the shipped CSVs (seconds):**

```bash
python3 scripts/make_figures.py --run final/ --out final/figs/
```

**Full final run (a few hours on a 10 core laptop):**

```bash
bash scripts/run_final.sh
```

The script runs the six recordings in parallel at 0.05 m grid resolution on
every frame, merges the CSVs, runs the mechanism analysis and draws the
figures. It writes:

- `final/q1_doseresponse.csv`: both estimators against injected dropout rate
- `final/q2_ksweep.csv`: sweep of the growth rate k
- `final/gaps.csv`: natural gap statistics per recording
- `final/mechanism.csv`: mechanism and keypoint yield analysis
- `final/figs/`: the two result figures

`run_final.sh` calls `caffeinate`, which exists only on macOS. On Linux,
remove `caffeinate -i` from the two lines that use it.

**Hold-last timeout sweep:**

```bash
for T in 1 2 5; do
  python3 scripts/run_paper_analysis.py --bags npz_replay_trim/*.npz \
    --config sweep_thold/cfg_$T.yaml --d-contact config/d_contact_tape.json \
    --out-prefix sweep_thold/t$T/ --skip-ksweep
done
```

**Ablations and static ground truth:**

```bash
for b in npz_replay_trim/310826_B*.npz; do
  n=$(basename "$b" .npz)
  python3 scripts/run_review_checks.py --bags "$b" --out "review/$n/"
done
```

Each recording gets an `ablations.csv`. The static recordings also get
`extents.json` and `static_ground_truth.csv`.

**Further checks (no growth, predicted keypoints, contact radii):**

```bash
python3 extra_checks/k0_static.py               # k = 0 against the static reference, occupied fraction
python3 extra_checks/k0_inject.py               # k = 0 under injected dropout
python3 extra_checks/check_predicted_frames.py  # share of keypoints predicted by the SDK
python3 extra_checks/check_contact_radii.py     # contact radii against segment half thickness
```

The outputs of our runs are stored next to the scripts.

**Contact radii from the tape measurements:**

```bash
python3 scripts/tape_to_dcontact.py   # reads config/tape_measurements.yaml
```

## Dropout injection protocol

What `pmp_risk_eval/occlusion_injector.py` and `scripts/run_paper_analysis.py` do in the final run.

**Configuration.** `config/eval_params_final.yaml`, block `paper`: `seed: 0`, `n_repeats: 5`,
`rho_grid: 0.0 to 0.9`, `rho_for_k_sweep: 0.0, 0.3, 0.6, 0.9`, `burst: auto`.

**Random numbers.** Each call of `run_paper_analysis.py` creates one generator,
`numpy.random.default_rng(0)`. `scripts/run_final.sh` calls the script once per recording, so every
recording starts from seed 0. The generator is drawn in a fixed order: the dose-response runs first
(independent mode, then structured; rates in ascending order; repeats 0 to 4), then the sweep of
the growth rate (k in ascending order; independent, then structured; rates ascending; repeats 0 to
4). Both methods are run on the same injected mask. Passing several recordings in one call, or in a
different order, changes the random stream.

**Episode length.** The median natural gap of that recording (`final/gaps.csv`, column
`burst_used_s`, 0.067 to 0.100 s) divided by the median frame interval and rounded, with a minimum
of one frame. This gives two or three frames here.

**Episode start.** Independent mode treats each keypoint separately. The frames are walked in
order. At a frame where the keypoint is seen, an episode starts with probability
p = rho / ((1 - rho) T + rho), where T is the episode length in frames. Structured mode does the
same for each of the four limb chains in `pmp_risk_eval/types.py` (`CHAINS`: shoulder to hand and
hip to foot, left and right). A chain is eligible when any of its keypoints is seen, and all of its
keypoints are removed together.

**Overlap.** Once an episode starts, the walk jumps to its end, so two episodes of the same keypoint
(or chain) never overlap. Episodes of different keypoints (or chains) are independent and may
coincide. An episode removes all T frames, including frames where the keypoint was already
missing, and is cut short by the end of the recording.

**Realised rate.** The share of originally seen keypoint readings that were removed (chain
keypoints only in structured mode), stored per run as `realised_rho`. It stayed within 0.025 of the
target in the final run.

**Run count.** A rate of zero is a single run with nothing injected. Per recording and method this
gives 2 x (1 + 9 x 5) = 92 dose-response runs and 8 x 2 x (1 + 3 x 5) = 256 sweep runs, 348 in
total, or 2088 over the six recordings.

**References.** Both gap-filled references are built once per recording from the uninjected
stream.

The two injection scripts in `extra_checks/` use `default_rng(12345)`, so their masks are separate from
the final run.

## Data

Each `.npz` file in `npz_replay_trim/` holds the skeleton stream of one
recording: frame times, and for every detected body its 38 keypoints
(ZED SDK BODY_38), a validity flag and a confidence value per keypoint, the
body id and the tracking state. There are no images and no depth.

Every detected body is stored, not only the manikin. In the dynamic
recordings an operator stands beside the bed, and that skeleton is in the
data. Subject selection happens at analysis time from the bed bounds in the
configuration.

The files were produced by replaying the recorded SVO2 files through the ZED
SDK on the lab computer. The raw recordings (ROS 2 bags and SVO2 files) are
not part of this bundle and are available on request.

## Other scripts

`run_paper_analysis.py`, `run_mechanism.py`, `run_review_checks.py`,
`make_figures.py`, `tape_to_dcontact.py` and `run_final.sh` are the ones the
paper uses. The remaining scripts belong to the earlier bag based pipeline or
to data export on the lab computer. `export_npz.py`, `check_bag.py` and
`pmp_risk_eval/bag_reader.py` need ROS 2 Humble with the ZED message types
when given a bag.

## Notes

- Parts of this code were written with assistance from Claude (Anthropic).
  All results were run and checked by the authors.
- Licence: the licence of the parent repository applies.
- Contact: Anupama Ginige, a.ginige@westernsydney.edu.au
