import json
import numpy as np
import pandas as pd
from paths import DATA, RESULTS, REFERENCE
from court_geometry import offset, JOIN, R
r=pd.read_csv(DATA/'shots.csv.gz');r=r[r.scope.eq('regular')].copy()
assert np.allclose(offset([R,0,JOIN],[0,22,22]),0,atol=1e-10)
assert np.allclose(offset([R-2,R+2,0,0],[0,0,20,24]),[-2,2,-2,2])
assert np.allclose(offset(r.x,r.y),offset(r.x,-r.y))
d=offset(r.x,r.y)
gx,gy=np.meshgrid(np.arange(-5.225,41.75,.05),np.arange(-24.975,25,.05));gd=offset(gx,gy)
rows=[];edges=np.arange(-4,4,1)
for name,mask,gmask in [('All locations',np.ones(len(r),bool),np.ones_like(gx,bool)),('Curved arc region',r.x.to_numpy()>JOIN,gx>JOIN),('Corner region',r.x.to_numpy()<=JOIN,gx<=JOIN)]:
 counts=np.histogram(d[mask],edges)[0];area=np.histogram(gd[gmask],edges)[0]*.0025
 for lo,hi,c,a in zip(edges[:-1],edges[1:],counts,area):rows.append({'region':name,'offset_lower':lo,'offset_upper':hi,'attempts':int(c),'area_sqft':a,'attempts_per_sqft':c/a})
table=pd.DataFrame(rows);old=pd.read_csv(REFERENCE/'arc_density.csv')
assert np.array_equal(table.attempts,old.attempts)
table.to_csv(RESULTS/'arc_density.csv',index=False)
counts=r[['game_id']].copy();counts['inside']=(d>=-2)&(d<0);counts['outside']=(d>0)&(d<=2)
inside_area=((gd>=-2)&(gd<0)).sum()*.0025;outside_area=((gd>0)&(gd<=2)).sum()*.0025
g=counts.groupby('game_id')[['inside','outside']].sum();rng=np.random.default_rng(2026)
w=rng.multinomial(len(g),np.repeat(1/len(g),len(g)),size=1000)
ratios=(w@g.outside.to_numpy()/outside_area)/(w@g.inside.to_numpy()/inside_area)
jitter=[]
for _ in range(100):
 jd=offset(np.clip(r.x+rng.uniform(-.5,.5,len(r)),-5.25,88.75),np.clip(r.y+rng.uniform(-.5,.5,len(r)),-25,25))
 jitter.append((((jd>0)&(jd<=2)).sum()/outside_area)/(((jd>=-2)&(jd<0)).sum()/inside_area))
result={'half_court_shots_shown':int(r.x.between(-5.25,41.75).sum()),'shots_outside_view':int((~r.x.between(-5.25,41.75)).sum()),'inside_2ft_attempts':int(g.inside.sum()),'outside_2ft_attempts':int(g.outside.sum()),'exactly_on_line_attempts_excluded_from_ratio':int(np.isclose(d,0).sum()),'inside_2ft_area':inside_area,'outside_2ft_area':outside_area,'outside_inside_density_ratio':(g.outside.sum()/outside_area)/(g.inside.sum()/inside_area),'game_bootstrap_95_interval':np.quantile(ratios,[.025,.975]).tolist(),'coordinate_sensitivity_ratio_range':[min(jitter),max(jitter)],'sensitivity_note':'Artificial independent uniform coordinate perturbations within half a foot; sensitivity scenarios, not inferred movement or a confidence interval.'}
(RESULTS/'arc_review.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
