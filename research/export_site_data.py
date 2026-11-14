from pathlib import Path
import json
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
from paths import SITE, DATA, RESULTS, REFERENCE
s=pd.read_csv(DATA/'shots.csv.gz');s=s[s.scope.eq('regular')].copy()
half=s[s.x.between(-5.25,41.75)]
a=half.groupby(['x','y','team_id','athlete_id_1']).agg(n=('made','size'),k=('made','sum'),points=('points','sum')).reset_index()
assert a.n.sum()==38441 and a.k.sum()==16926 and a.points.sum()==38533
names=s.groupby('athlete_id_1').player.first()
teams=pd.read_csv(REFERENCE/'team_summary.csv')
gp=pd.read_csv(REFERENCE/'gp_surface.csv')
metrics=pd.read_csv(RESULTS/'temporal_metrics.csv')
cv=pd.read_csv(RESULTS/'spatial_cv.csv')
cvsum=[]
for (buffer,model),b in cv.groupby(['buffer_ft','model']):
 cvsum.append({'buffer':int(buffer),'model':model,'brier':float(np.average(b.brier,weights=b.shots)),'log_loss':float(np.average(b.log_loss,weights=b.shots))})
base=RESULTS
data={'cells':a.astype(int).values.tolist(),'players':{str(int(k)):v for k,v in names.items()},'teams':{str(int(r.team_id)):r.team for _,r in teams.iterrows()},'gp':np.round(gp[['x','y','mean','lower','upper']].to_numpy(),5).tolist(),'arc':json.loads((RESULTS/'arc_review.json').read_text()),'arcBins':pd.read_csv(RESULTS/'arc_density.csv').query("region == 'All locations'").to_dict('records'),'metrics':metrics.to_dict('records'),'spatialCV':cvsum,'folds':pd.read_csv(base/'spatial_folds.csv').to_dict('records'),'bootstrap':pd.read_csv(base/'game_bootstrap.csv').to_dict('records'),'mesh':json.loads((base/'mesh.json').read_text()),'variogram':pd.read_csv(base/'Universal_kriging_variogram.csv').to_dict('records'),'source':json.loads((RESULTS/'source_audit.json').read_text())}
train=half[half.date.lt('2025-08-10')][['x','y']].drop_duplicates().copy()
train['fold']=(train.x.ge(16).astype(int)*2+train.y.ge(0).astype(int))
train['nearest']=cKDTree(train[train.fold.eq(0)][['x','y']]).query(train[['x','y']])[0]
data['trainingLocations']=np.round(train.to_numpy(),5).tolist()
data['kriging']=json.loads((RESULTS/'model_parameters.json').read_text())['Universal kriging']
(SITE/'assets/data.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False))
print('Exported',len(a),'cell/player/team groups')
