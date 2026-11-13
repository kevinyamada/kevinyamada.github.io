import json
import numpy as np
import pandas as pd
from paths import DATA, RESULTS
s=pd.read_csv(DATA/'shots.csv.gz');p=pd.read_csv(DATA/'predictions.csv.gz')
a=p.merge(s[['game_id','game_play_number','made','points','points_attempted','scope','date','x','y']],on=['game_id','game_play_number'],validate='one_to_one',suffixes=['','_source'])
assert len(a)==len(p) and a.made.eq(a.made_source).all()
models=[c for c in p if c not in ['game_id','game_play_number','made','split']]
rows=[];calibration=[];by_value=[]
metrics=pd.read_csv(RESULTS/'temporal_metrics.csv')
for split,b in a.groupby('split'):
 for name in models:
  q=b[name].clip(1e-15,1-1e-15);y=b.made
  row=metrics[(metrics.split==split)&(metrics.model==name)].iloc[0]
  assert np.isclose(np.mean((y-q)**2),row.brier)
  assert np.isclose(-np.mean(y*np.log(q)+(1-y)*np.log(1-q)),row.log_loss)
  xp=q*b.points_attempted
  rows.append({'split':split,'model':name,'shots':len(b),'expected_points_mean':xp.mean(),'observed_points_mean':b.points.mean(),'rmse':np.sqrt(np.mean((xp-b.points)**2)),'mae':np.mean(abs(xp-b.points)),'condition':'Known two or three point attempt value; field goal points only'})
  c=b.assign(bin=pd.cut(q,np.linspace(0,1,11),include_lowest=True))
  for key,g in c.groupby('bin',observed=True):calibration.append({'split':split,'model':name,'bin':str(key),'shots':len(g),'mean_prediction':g[name].mean(),'observed_rate':g.made.mean()})
  for value,g in b.groupby('points_attempted'):by_value.append({'split':split,'model':name,'attempt_value':int(value),'shots':len(g),'mean_prediction':g[name].mean(),'observed_rate':g.made.mean(),'brier':np.mean((g[name]-g.made)**2)})
pd.DataFrame(rows).to_csv(RESULTS/'expected_points_metrics.csv',index=False)
pd.DataFrame(calibration).to_csv(RESULTS/'calibration.csv',index=False)
pd.DataFrame(by_value).to_csv(RESULTS/'shot_value_calibration.csv',index=False)
folds=pd.read_csv(RESULTS/'spatial_folds.csv');assert (folds.minimum_separation_ft>folds.buffer_ft).all()
cv=pd.read_csv(RESULTS/'spatial_cv.csv');assert cv.groupby(['buffer_ft','model']).shots.sum().eq(26686).all()
r=s[(s.scope=='regular')&s.x.between(-5.25,41.75)]
t=r[r.date<'2025-08-10'];v=r[(r.date>='2025-08-10')&(r.date<'2025-08-27')];q=r[r.date>='2025-08-27']
assert not set(t.game_id)&set(v.game_id) and not set(v.game_id)&set(q.game_id)
a=s[(s.scope=='regular')&(s.x==2)&(s.y==0)&(s.team_id==5)]
assert len(a)==51 and a.made.sum()==31 and a.points.sum()==62
b=a.groupby('player').points.sum();assert b.idxmax()=='Natasha Howard' and b.max()==16
checks={'manual_brier_and_log_loss_reproduced':True,'prediction_labels_match_source_event_keys':True,'temporal_games_disjoint':True,'all_spatial_separations_exceed_buffer':True,'all_spatial_folds_partition_training_shots':True,'interactive_indiana_example_matches_raw_rows':True,'modal_dollars_spent':0}
(RESULTS/'verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
print(pd.DataFrame(rows).query("split=='test'").sort_values('rmse').to_string(index=False))
