"""Camera-ready checks requested by the ACRA reviewers.

Further occupancy policies, run with the final configuration (same subject
filter, confidence gate, grid, contact radii), every frame, nothing injected
unless --rho is given.

  hold_0.5        hold-last, timeout 0.5 s (paper baseline, reproduced)
  hold_m10_0.5    hold-last with a fixed 0.10 m safety margin, timeout 0.5 s
  cv_0.5          constant-velocity prediction (velocity fitted over the last
                  0.3 s of sightings), radius d_contact, deleted after 0.5 s.
                  The keypoint analogue of a Kalman track with a maximum age.
  keep_k0         keep every keypoint, no growth (paper Table 3 row)
  grow_t2         growing radius, but deleted after 2 s (a horizon cap)
  grow            growing radius, k = 0.8 (paper method, reproduced)
  grow_cap        growing radius, with every lost element also limited to a
                  ball around the nearest keypoint seen in this frame, of
                  radius equal to the summed bone lengths between them plus
                  the contact radius (a kinematic chain cap)

Bone lengths for the cap are the 95th percentile of the tracked distance
between the two ends of each bone when both are seen, pooled over the six
recordings. Bones never seen with both ends get no cap.

Outputs (extra_checks/):
  camera_ready_variants.json   per recording and policy: occupied fraction,
                               falls, violations (endpoint rule as in the
                               paper, and counting the cap anchor), static
                               reference misses, gap filled reference misses
  camera_ready_bones.json      bone lengths used by the cap

    python3 extra_checks/camera_ready_variants.py            # rho = 0
    python3 extra_checks/camera_ready_variants.py --rho 0.6  # injected
"""
import sys, os, json, glob, time, argparse
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from pmp_risk_eval import config_io, metrics
from pmp_risk_eval.subject_filter import SubjectFilter
from pmp_risk_eval.query_set import bed_grid, grid_bounds
from pmp_risk_eval.reference_builder import reference_occupancy
from pmp_risk_eval.occlusion_injector import (inject_independent,
                                              natural_gap_stats, realised_rate)
from pmp_risk_eval.occupancy_field import _point_segment_distance
from pmp_risk_eval.types import N_KP, EDGES
from run_paper_analysis import load_bag
from run_review_checks import gt_occupancy, cover

cfg = config_io.load('config/eval_params_final.yaml')
pp = cfg['paper']
bb = config_io.bed_bounds(cfg)
D = np.array(json.load(open('config/d_contact_tape.json'))['d_contact'])
Q = bed_grid(grid_bounds(bb, pp.get('grid_margin', 0.0)), res=pp['grid_res'])
E = np.asarray(EDGES, dtype=int)
BAGS = sorted(glob.glob('npz_replay_trim/310826_B*.npz'))

POLICIES = {
    'hold_0.5':     dict(kind='hold', t_hold=0.5),
    'hold_m10_0.5': dict(kind='hold', t_hold=0.5, margin=0.10),
    'cv_0.5':       dict(kind='cv', t_hold=0.5),
    'keep_k0':      dict(kind='grow', k=0.0),
    'grow_t2':      dict(kind='grow', k=0.8, t_hold=2.0),
    'grow':         dict(kind='grow', k=0.8),
    'grow_cap':     dict(kind='grow', k=0.8, cap=True),
}


def _sel(bag):
    return SubjectFilter('bed', bed_bounds=bb, hysteresis=pp['hysteresis'],
                         seed=pp.get('subject_seed'))


def bone_lengths():
    L = np.full(len(E), np.inf)
    dists = [[] for _ in E]
    for bag in BAGS:
        _, pos, val = load_bag(bag, _sel(bag), pp['kp_conf_min'])
        for e, (i, j) in enumerate(E):
            m = val[:, i] & val[:, j]
            if m.any():
                dists[e].extend(np.linalg.norm(pos[m, i] - pos[m, j], axis=1))
    n = np.array([len(x) for x in dists])
    for e in range(len(E)):
        if n[e] >= 10:
            L[e] = float(np.percentile(dists[e], 95))
    return L, n


def path_lengths(L):
    """All-pairs path length on the skeleton tree (inf if a bone has no
    length estimate)."""
    P = np.full((N_KP, N_KP), np.inf)
    np.fill_diagonal(P, 0.0)
    for e, (i, j) in enumerate(E):
        P[i, j] = P[j, i] = L[e]
    for m in range(N_KP):
        P = np.minimum(P, P[:, m:m + 1] + P[m:m + 1, :])
    return P


class Policy:
    def __init__(self, kind, k=0.8, t_hold=None, margin=0.0, cap=False,
                 P=None, v_win=0.3):
        self.kind, self.k, self.t_hold = kind, k, t_hold
        self.margin, self.cap, self.P, self.v_win = margin, cap, P, v_win
        self.pos = np.full((N_KP, 3), np.nan)
        self.last_t = np.full(N_KP, np.nan)
        self.ever = np.zeros(N_KP, bool)
        self.vel = np.zeros((N_KP, 3))
        self.hist = [[] for _ in range(N_KP)]

    def update(self, t, x, seen):
        for i in np.where(seen)[0]:
            self.pos[i] = x[i]
            self.last_t[i] = t
            self.ever[i] = True
            if self.kind == 'cv':
                h = self.hist[i]
                h.append((t, x[i].copy()))
                while h and h[0][0] < t - self.v_win:
                    h.pop(0)
                if len(h) >= 3 and h[-1][0] - h[0][0] >= 0.1:
                    ts = np.array([a for a, _ in h])
                    xs = np.array([b for _, b in h])
                    tc = ts - ts.mean()
                    self.vel[i] = (tc[:, None] * (xs - xs.mean(0))).sum(0) / (tc ** 2).sum()
                else:
                    self.vel[i] = 0.0
        self.t = t
        self.seen = seen.copy()

    def elements(self):
        """Balls and tubes with optional cap balls. Returns dict of arrays,
        or None when nothing is active."""
        age = np.where(self.ever, self.t - self.last_t, np.inf)
        active = self.ever & np.isfinite(self.pos).all(1)
        if self.t_hold is not None:
            active &= age <= self.t_hold
        if not active.any():
            return None
        a0 = np.where(np.isfinite(age), age, 0.0)
        x = self.pos.copy()
        if self.kind == 'cv':
            x = x + self.vel * a0[:, None]
        r = D + self.margin
        if self.kind == 'grow':
            r = r + self.k * a0
        balls = np.where(active)[0]
        both = active[E[:, 0]] & active[E[:, 1]]
        tubes = E[both]
        n_b, n_t = len(balls), len(tubes)
        ends = np.concatenate([np.stack([balls, balls], 1), tubes]) if n_t else np.stack([balls, balls], 1)
        # chain cap: element e with an unseen end is limited, for every
        # keypoint a seen before, to the ball around a's stored position of
        # radius (summed bone lengths a..far end) + k * age_a + contact radius.
        # Between observations every such ball only grows.
        capR = None
        if self.cap:
            anc = self.ever & np.isfinite(self.pos).all(1)
            lost = ~(self.seen[ends[:, 0]] & self.seen[ends[:, 1]])
            # a frame with nothing seen brings no evidence: an element that
            # was uncapped stays uncapped until something is seen again
            prev = getattr(self, '_lost_prev', None)
            if not self.seen.any() and prev is not None and len(prev) == len(lost):
                lost = prev
            self._lost_prev = lost
            if anc.any() and lost.any():
                A = np.where(anc)[0]
                pl = np.maximum(self.P[A][:, ends[:, 0]], self.P[A][:, ends[:, 1]])  # (A,E)
                dmax = np.maximum(D[ends[:, 0]], D[ends[:, 1]])
                R = pl + self.k * a0[A][:, None] + dmax[None, :]
                R[:, ~lost] = np.inf
                capR = (A, R)
        return dict(x=x, r=r, balls=balls, tubes=tubes, ends=ends, capR=capR,
                    xa=self.pos.copy())

    def covered(self, el, q, keep=None):
        """keep: (n_el,) bool element mask. Returns (N,) bool."""
        if el is None:
            fill = self.kind == 'grow' and self.t_hold is None
            return np.full(len(q), fill)
        x, r, balls, tubes = el['x'], el['r'], el['balls'], el['tubes']
        n_b = len(balls)
        n_el = len(el['ends'])
        keep = np.ones(n_el, bool) if keep is None else keep
        inside_cap = np.ones((len(q), n_el), bool)
        if el['capR'] is not None:
            A, R = el['capR']
            capped = np.isfinite(R).any(0) & keep
            if capped.any():
                dq = np.linalg.norm(q[:, None, :] - el['xa'][A][None], axis=2)  # (N,A)
                inside_cap[:, capped] = (dq[:, :, None] <= R[:, capped][None]).all(1)
        c = np.zeros(len(q), bool)
        kb = keep[:n_b]
        if kb.any():
            db = np.linalg.norm(q[:, None, :] - x[balls[kb]][None], axis=2)
            c |= ((db <= r[balls[kb]][None, :]) & inside_cap[:, :n_b][:, kb]).any(1)
        if len(tubes):
            kt = keep[n_b:]
            if kt.any():
                i, j = tubes[kt, 0], tubes[kt, 1]
                ds = _point_segment_distance(q, x[i], x[j])
                c |= ((ds <= np.maximum(r[i], r[j])[None, :])
                      & inside_cap[:, n_b:][:, kt]).any(1)
        return c


def run(times, pos, val, spec, P):
    pol = Policy(P=P, **spec)
    U = np.zeros((len(times), len(Q)), np.uint8)
    falls = viol = viol_anchor = 0
    prev_el = None
    for f, t in enumerate(times):
        pol.update(t, pos[f], val[f])
        el = pol.elements()
        U[f] = pol.covered(el, Q)
        if f and prev_el is not None:
            fell = (U[f - 1] == 1) & (U[f] == 0)
            if fell.any():
                falls += int(fell.sum())
                unseen = ~val[f]
                k1 = unseen[prev_el['ends']].all(1)
                v1 = pol.covered(prev_el, Q[fell], keep=k1)
                viol += int(v1.sum())
                # relaxed rule for the cap: any keypoint seen counts as evidence
                k2 = k1 & (not val[f].any()) if spec.get('cap') else k1
                v2 = pol.covered(prev_el, Q[fell], keep=k2)
                viol_anchor += int(v2.sum())
        prev_el = el
    return U, dict(falls=falls, violations=viol, violations_anchor=viol_anchor)


def static_misses(name, times, pos, val, U):
    x_gt, real = gt_occupancy(pos, val, D, Q, 30)
    tr_seen = np.cumsum(val, axis=0) > 0
    miss = tot = 0
    cache = {}
    for f in range(len(times)):
        act = real & tr_seen[f]
        if not act.any():
            continue
        key = act.tobytes()
        if key not in cache:
            cache[key] = cover(Q, x_gt, act, D)
        g = cache[key]
        tot += int(g.sum())
        miss += int((g & (U[f] == 0)).sum())
    return miss, tot


def one(args):
    bag, rho, P, reps = args
    name = os.path.basename(bag).replace('.npz', '').replace('310826_', '')
    times, pos, val = load_bag(bag, _sel(bag), pp['kp_conf_min'])
    refs = {s: reference_occupancy(times, pos, val, D, Q, scheme=s,
                                   max_gap=pp['max_gap_ref'])
            for s in ('linear', 'hold')}
    burst = natural_gap_stats(times, val)['median_s']
    rng = np.random.default_rng(2026)
    rows = []
    for rep in range(reps if rho > 0 else 1):
        V = val if rho == 0 else inject_independent(times, val, rho, burst, rng=rng)
        rr = realised_rate(val, V)
        for lab, spec in POLICIES.items():
            if rho > 0 and lab in ('grow_t2', 'keep_k0'):
                continue
            t0 = time.time()
            U, mono = run(times, pos, V, spec, P)
            m = metrics.free_space_error_agreed(U, refs['linear'], refs['hold'])
            row = dict(bag=name, policy=lab, rho=rho, rep=rep,
                       realised=round(rr, 4), frames=len(times),
                       points=len(Q), occ=float(U.mean()),
                       fse_gap=m['free_space_error_agreed_given_ref'],
                       gap_missed=int(round(m['free_space_error_agreed_given_ref'] * m['n_ref_agreed'])),
                       gap_ref=int(m['n_ref_agreed']), **mono)
            if 'static' in name and rho == 0:
                row['static_missed'], row['static_ref'] = static_misses(
                    name, times, pos, val, U)
            rows.append(row)
            print(f"{name:<24} {lab:<13} rho {rho} rep {rep} occ {row['occ']:.3f}"
                  f" viol {mono['violations']} (anchor {mono['violations_anchor']})"
                  f" fse_gap {100*row['fse_gap']:.3f}%"
                  + (f" static {100*row['static_missed']/row['static_ref']:.2f}%"
                     if 'static_ref' in row else '')
                  + f" ({time.time()-t0:.0f}s)", flush=True)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rho', type=float, default=0.0)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    t0 = time.time()
    L, n = bone_lengths()
    json.dump(dict(edges=E.tolist(), length_p95_m=[None if not np.isfinite(v) else v for v in L],
                   n_pairs=n.tolist()),
              open('extra_checks/camera_ready_bones.json', 'w'), indent=1)
    P = path_lengths(L)
    from multiprocessing import Pool
    with Pool(a.procs) as p:
        res = p.map(one, [(b, a.rho, P, a.reps) for b in BAGS if not a.only or any(o in b for o in a.only)], chunksize=1)
    out = f"extra_checks/camera_ready_variants{'' if a.rho == 0 else '_rho' + str(a.rho)}{a.tag}.json"
    json.dump([r for rs in res for r in rs], open(out, 'w'), indent=1)
    print('wrote', out, 'seconds', round(time.time() - t0))
