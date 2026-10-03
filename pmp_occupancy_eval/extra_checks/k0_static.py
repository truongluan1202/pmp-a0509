"""k = 0 (keep every keypoint, no growth) against the static reference and for occupied fraction.
Uses the package's own functions and the final configuration."""
import sys, os, json, glob, time
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')  # run from the package root: python3 extra_checks/<name>.py
import numpy as np
from pmp_risk_eval import config_io
from pmp_risk_eval.subject_filter import SubjectFilter
from pmp_risk_eval.query_set import bed_grid, grid_bounds
from run_paper_analysis import load_bag
from run_review_checks import walk, gt_occupancy, cover
cfg=config_io.load('config/eval_params_final.yaml'); pp=cfg['paper']; bb=config_io.bed_bounds(cfg)
d=np.array(json.load(open('config/d_contact_tape.json'))['d_contact'])
Q=bed_grid(grid_bounds(bb,pp.get('grid_margin',0.0)),res=pp['grid_res'])
def one(bag):
    name=os.path.basename(bag).replace('.npz','').replace('310826_','')
    sf=SubjectFilter('bed',bed_bounds=bb,hysteresis=pp['hysteresis'],seed=pp.get('subject_seed'))
    times,pos,val=load_bag(bag,sf,pp['kp_conf_min'])
    out={'bag':name}
    runs={}
    for lab,arm,k,th in (('grow_k0.8','proposed',0.8,0.5),('keep_k0','proposed',0.0,0.5),('hold_0.5','hold',0.8,0.5)):
        occ,U=walk(times,pos,val,d,Q,arm,k,t_hold=th,keep_U=True)
        runs[lab]=U; out['occ_'+lab]=float(occ.mean())
    if 'static' in name:
        x_gt,real=gt_occupancy(pos,val,d,Q,30); tr_seen=np.cumsum(val,axis=0)>0
        miss={k:0 for k in runs}; tot=0; cache={}
        for f in range(len(times)):
            act=real&tr_seen[f]
            if not act.any(): continue
            key=act.tobytes()
            if key not in cache: cache[key]=cover(Q,x_gt,act,d)
            g=cache[key]; tot+=int(g.sum())
            for k,U in runs.items(): miss[k]+=int((g&(U[f]==0)).sum())
        for k in runs: out['missed_'+k]=miss[k]
        out['gt_point_frames']=tot
    return out
if __name__=='__main__':
    from multiprocessing import Pool
    bags=sorted(glob.glob('npz_replay_trim/310826_B*.npz'))
    t0=time.time()
    with Pool(min(6,os.cpu_count())) as p: res=p.map(one,bags)
    json.dump(res,open('extra_checks/k0_static.json','w'),indent=1)
    for r in res: print(r)
    print('seconds',round(time.time()-t0))
