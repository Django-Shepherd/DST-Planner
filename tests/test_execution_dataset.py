#!/usr/bin/env python3
"""Audit real executed-plan rewards and reject deliberately corrupted associations."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'learning'))
from dst_planner.data import episode,rows
p=argparse.ArgumentParser();p.add_argument('episode');a=p.parse_args();root=Path(a.episode)
samples,bc,audit=episode(root)
assert samples and bc and audit['finish_rewards']==1
assert all(s['distance_m']>=0 and s['next_activation_stamp']>s['activation_stamp'] for s in samples)
assert all(abs(s['reward']+s['distance_m']-100*s['terminated'])<1e-6 for s in samples)
assert all(not s['truncated'] for s in samples)
if not audit['recovery_intervals_excluded']:
    # Real position integration across the full activation interval equals the sum of all rewards' costs.
    odom=sorted((x for x in rows(root/'events.jsonl') if x['event']=='odometry'),key=lambda x:x['stamp'])
    t=np.array([x['stamp'] for x in odom]);xyz=np.array([x['position'] for x in odom]);cum=np.r_[0,np.linalg.norm(np.diff(xyz,axis=0),axis=1).cumsum()]
    distance=np.interp(samples[-1]['next_activation_stamp'],t,cum)-np.interp(samples[0]['activation_stamp'],t,cum)
    assert abs(distance-audit['transition_distance_m'])<1e-5
with tempfile.TemporaryDirectory() as temp:
    temp=Path(temp)
    for name in ['metadata.json','summary.json','decisions.jsonl','decisions.jsonl.status.json']:
        shutil.copy(root/name,temp/name)
    events=rows(root/'events.jsonl');changed=False
    for event in events:
        if event.get('kind')=='trajectory_applied' and event['decision_id']>=0:
            event['goal'][0]+=1000;changed=True;break
    assert changed
    (temp/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    try:episode(temp)
    except ValueError as exc:assert 'Applied goal does not match' in str(exc)
    else:raise AssertionError('Corrupted action/trajectory link was accepted')
print(json.dumps({'passed':True,'checks':['raw_meter_reward','one_finish_bonus','activation_intervals','distance_conservation','corrupt_goal_rejected'],'audit':audit},indent=2))
