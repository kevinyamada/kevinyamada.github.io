import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss,brier_score_loss
from spatial_methods import SPDE
from paths import DATA, RESULTS
OUT=RESULTS
s=pd.read_csv(DATA/'shots.csv.gz');s=s[s.scope.eq('regular')&s.x.between(-5.25,41.75)]
a=s[s.date.lt('2025-08-10')];b=s[s.date.ge('2025-08-10')&s.date.lt('2025-08-27')]
rows=[];last=None
for spacing in [3,2,1.5]:
 m=SPDE(16,.3,spacing).fit(a);p=m.predict(b)
 rows.append({'spacing_ft':spacing,'nodes':len(m.nodes),'validation_log_loss':log_loss(b.made,p),'validation_brier':brier_score_loss(b.made,p),'mean_change_from_previous':None if last is None else float(abs(last-p).mean()),'maximum_change_from_previous':None if last is None else float(abs(last-p).max())});last=p
 print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(OUT/'mesh_refinement.csv',index=False)
