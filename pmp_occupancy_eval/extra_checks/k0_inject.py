"""k = 0 under injected dropout (rho = 0.6, independent, 5 repeats), same masks also run at k = 0.8 as a control.
Same configuration, references and metric as the final run (free_space_error_agreed_given_ref)."""
import sys, os, json, glob, time
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')  # run from the package root: python3 extra_checks/<name>.py
import numpy as np
from pmp_risk_eval import config_io, metrics
from pmp_risk_eval.subject_filter import SubjectFilter
from pmp_risk_eval.query_set import bed_grid, grid_bounds
from pmp_risk_eval.reference_builder import reference_occupancy
from pmp_risk_eval.occlusion_injector import inject_independent, natural_gap_stats, realised_rate
from run_paper_analysis import load_bag, run_arm
from run_review_checks import walk
cfg=config_io.load('config/eval_params_final.yaml'); pp=cfg['paper']; bb=config_io.bed_bounds(cfg)
d=np.array(json.load(open('config/d_contact_tape.json'))['d_contact'])
Q=bed_grid(grid_bounds(bb,pp.get('grid_margin',0.0)),res=pp['grid_res'])
def one(bag):
    name=os.path.basename(bag).replace('.npz','').replace('310826_','')
    sf=SubjectFilter('bed',bed_bounds=bb,hysteresis=pp['hysteresis'],seed=pp.get('subject_seed'))
    times,pos,val=load_bag(bag,sf,pp['kp_conf_min'])
    burst=natural_gap_stats(times,val)['median_s']
    refs={s:reference_occupancy(times,pos,val,d,Q,scheme=s,max_gap=pp['max_gap_ref']) for s in ('linear','hold')}
    rows=[]
    U,mono=run_arm(times,pos,val,d,Q,'proposed',0.0,pp['r0'],pp['t_hold'])
    m=metrics.free_space_error_agreed(U,refs['linear'],refs['hold'])
    rows.append(dict(bag=name,k=0.0,rho=0.0,rep=0,fse=m['free_space_error_agreed_given_ref'],occ=float(U.mean()),violations=int(mono['violations']),falls=int(mono['falls'])))
    rng=np.random.default_rng(12345)
    for rep in range(5):
        V=inject_independent(times,val,0.6,burst,rng=rng)
        rate=realised_rate(val,V)
        for k in (0.0,0.8):
            occ,U=walk(times,pos,V,d,Q,'proposed',k,keep_U=True)
            m=metrics.free_space_error_agreed(U,refs['linear'],refs['hold'])
            rows.append(dict(bag=name,k=k,rho=0.6,rep=rep,realised=rate,fse=m['free_space_error_agreed_given_ref'],occ=float(occ.mean())))
        json.dump(rows,open(f'extra_checks/k0_inject_{name}.json','w'),indent=1)
    return rows
if __name__=='__main__':
    from multiprocessing import Pool
    bags=sorted(glob.glob('npz_replay_trim/310826_B*.npz'))
    t0=time.time()
    with Pool(2) as p: res=p.map(one,bags,chunksize=1)
    json.dump([r for rs in res for r in rs],open('extra_checks/k0_inject.json','w'),indent=1)
    print('seconds',round(time.time()-t0))
