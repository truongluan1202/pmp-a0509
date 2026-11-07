"""Camera-ready checks requested by the ACRA reviewers (no occupancy runs).

  A. Second person in the bed volume: in each recording, frames with more
     than one detected body, frames with more than one body whose torso lies
     in the bed volume, and how close the other body came to the manikin.
  B. Tracker jitter on the still manikin: spread of each keypoint about its
     median position, frame differenced speed (as in the paper's 0.34 to
     0.87 m/s) and the same speed from a line fitted over 0.3 s of sightings.
  C. Pause rule: share of frames in which the oldest keypoint is older than
     a_max, i.e. how often a controller that halts and waits for a fresh view
     would be halted. With all keypoints, and without finger and toe
     keypoints.

    python3 extra_checks/camera_ready_misc.py > extra_checks/camera_ready_misc.txt
"""
import sys, os, json, glob
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from pmp_risk_eval import config_io, npz_corpus, bag_reader
from pmp_risk_eval.subject_filter import SubjectFilter
from pmp_risk_eval.types import KP_NAMES, TORSO, N_KP
from run_paper_analysis import load_bag

cfg = config_io.load('config/eval_params_final.yaml')
pp = cfg['paper']
bb = config_io.bed_bounds(cfg)
BAGS = sorted(glob.glob('npz_replay_trim/310826_B*.npz'))
FINGERS = [i for i, n in enumerate(KP_NAMES) if 'HAND_' in n]
TOES = [i for i, n in enumerate(KP_NAMES) if 'TOE' in n]
DISTAL = [i for i, n in enumerate(KP_NAMES) if any(
    t in n for t in ('WRIST', 'ANKLE', 'HAND_', 'TOE', 'HEEL'))]


def sel():
    return SubjectFilter('bed', bed_bounds=bb, hysteresis=pp['hysteresis'],
                         seed=pp.get('subject_seed'))


def in_bed(body, m=0.0):
    v = body.valid[TORSO]
    if not v.any():
        return False
    c = np.nanmean(body.keypoints[TORSO][v], axis=0)
    return all(lo - m <= ci <= hi + m for ci, (lo, hi) in zip(c, bb))


print('A. second person')
for bag in BAGS:
    name = os.path.basename(bag)[7:-4]
    sf = sel()
    n = multi = multi_bed = 0
    dmin = []
    for fr in npz_corpus.read_frames_npz(bag):
        n += 1
        subj, _ = sf.select(fr)
        others = [b for b in fr.bodies if subj is None or b is not subj]
        if len(fr.bodies) > 1:
            multi += 1
        if sum(in_bed(b, pp['hysteresis']) for b in fr.bodies) > 1:
            multi_bed += 1
        if subj is not None and others:
            ps = subj.keypoints[subj.valid]
            for o in others:
                po = o.keypoints[o.valid]
                if len(ps) and len(po):
                    dmin.append(np.min(np.linalg.norm(ps[:, None] - po[None], axis=2)))
    dm = np.array(dmin)
    print(f'  {name:<24} frames {n:5d}  >1 body {multi:5d} ({100*multi/n:4.1f}%)'
          f'  >1 body in bed volume {multi_bed:4d} ({100*multi_bed/n:4.1f}%)'
          + (f'  other body nearest keypoint: median {np.median(dm):.2f} m,'
             f' min {dm.min():.2f} m, frames < 0.3 m {(dm < 0.3).sum()}' if len(dm) else '')
          + f'  ID changes {sf.report()["n_id_changes"]}')

print('\nB. jitter on the still manikin (keypoints seen in >= 30 frames)')
for bag in BAGS:
    if 'static' not in bag:
        continue
    name = os.path.basename(bag)[7:-4]
    t, pos, val = load_bag(bag, sel(), pp['kp_conf_min'])
    spread, vdiff, vfit, vdiff_d, vfit_d = [], [], [], [], []
    for i in range(N_KP):
        idx = np.where(val[:, i])[0]
        if len(idx) < 30:
            continue
        x = pos[idx, i]
        spread.append(np.sqrt(np.mean(np.sum((x - np.median(x, 0)) ** 2, 1))))
        cons = np.where(np.diff(idx) == 1)[0]
        vd = np.linalg.norm(x[cons + 1] - x[cons], axis=1) / np.diff(t[idx])[cons]
        vf = []
        for a in range(len(idx)):
            w = (t[idx] >= t[idx[a]] - 0.3) & (t[idx] <= t[idx[a]])
            if w.sum() >= 3 and t[idx[a]] - t[idx][w].min() >= 0.1:
                tc = t[idx][w] - t[idx][w].mean()
                xc = x[w] - x[w].mean(0)
                vf.append(np.linalg.norm((tc[:, None] * xc).sum(0) / (tc ** 2).sum()))
        vdiff += list(vd); vfit += vf
        if i in DISTAL:
            vdiff_d += list(vd); vfit_d += vf
    sp = np.array(spread)
    print(f'  {name:<24} position spread (RMS about median): median {1000*np.median(sp):.0f} mm,'
          f' range {1000*sp.min():.0f} to {1000*sp.max():.0f} mm')
    print(f'  {"":<24} hand and foot speed p95: frame differenced {np.percentile(vdiff_d,95):.2f} m/s,'
          f' fitted over 0.3 s {np.percentile(vfit_d,95):.2f} m/s'
          f'  (all keypoints {np.percentile(vdiff,95):.2f} / {np.percentile(vfit,95):.2f})')

print('\nC. pause rule: share of frames with oldest keypoint age > a_max')
res = {}
for bag in BAGS:
    name = os.path.basename(bag)[7:-4]
    t, pos, val = load_bag(bag, sel(), pp['kp_conf_min'])
    out = []
    for lab, drop in (('all', []), ('no fingers or toes', FINGERS + TOES)):
        v = val.copy()
        v[:, drop] = False
        last = np.full(N_KP, np.nan)
        oldest = np.zeros(len(t))
        for f in range(len(t)):
            last[v[f]] = t[f]
            ages = t[f] - last
            oldest[f] = np.nanmax(ages) if np.isfinite(ages).any() else np.inf
        out.append((lab, [float(np.mean(oldest > a)) for a in (0.5, 1.0, 2.0)]))
    res[name] = out
    print(f'  {name:<24} ' + '   '.join(
        f'{lab}: ' + ', '.join(f'>{a}s {100*s:4.1f}%' for a, s in zip((0.5, 1, 2), sh))
        for lab, sh in out))
