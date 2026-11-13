import hashlib
import json
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
from paths import DATA, RESULTS, REFERENCE
from court_geometry import offset, zones
sources={
 'pbp_2025.parquet':('https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_wnba_pbp/play_by_play_2025.parquet','aef69113f8362ad0623c3ae03ce447fe73e10c22a0f6223a9849448e0a3fd1d9'),
 'team_box_2025.parquet':('https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_wnba_team_boxscores/team_box_2025.parquet','b241743395be6808821a05134d2ea9386a585e6d60c3aa21dfe7c70f7893bf09')}
for name,(url,sha) in sources.items():
 p=DATA/name
 if not p.exists():urlretrieve(url,p)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,'Source changed; review the new release before replacing the expected hash'
d=pd.read_parquet(DATA/'pbp_2025.parquet').sort_values(['game_id','game_play_number'])
assert not d.duplicated(['game_id','game_play_number']).any()
for col in ['home_score','away_score']:d['before_'+col]=d.groupby('game_id')[col].shift().fillna(0)
d=d.sort_index()
s=d[d.shooting_play & ~d.type_text.str.contains('free throw',case=False,na=False)].copy()
s['made']=s.scoring_play.astype(int);s['points']=s.score_value.astype(int)
s['home']=s.team_id.eq(s.home_team_id).astype(int)
sign=np.where(s.home.eq(1),1,-1)
s['x']=41.75-s.coordinate_x*sign;s['y']=s.coordinate_y*sign
s['distance']=np.hypot(s.x,s.y);s['angle']=np.arctan2(abs(s.y),s.x)
s['scope']=np.where(s.game_id.isin([401781604,401736430]),'special',np.where(s.season_type.eq(3),'playoffs','regular'))
s['date']=pd.to_datetime(s.game_date).dt.strftime('%Y-%m-%d')
c=s.clock_display_value.str.split(':',expand=True).astype(float);s['seconds']=c[0]*60+c[1]
s['region']=zones(s.x,s.y);s['arc_offset_exact']=offset(s.x,s.y)
s['athlete_id_1']=s.athlete_id_1.astype(int);s['team_id']=s.team_id.astype(int)
players=pd.read_csv(REFERENCE/'player_summary.csv').set_index('athlete_id_1')
s['player']=s.athlete_id_1.map(players.name).fillna(s.athlete_name_1)
reg=s[s.scope.eq('regular')];checks={}
old=pd.read_csv(REFERENCE/'box_score_reconciliation.csv')
a=s.groupby(['game_id','team_id']).agg(fga=('made','size'),fgm=('made','sum'),fg_points=('points','sum')).reset_index().merge(old[['game_id','team_id','fga','fgm','fg_points']],on=['game_id','team_id'],suffixes=['','_old'],validate='one_to_one')
checks['game_team_totals_match']=bool(len(a)==624 and all(a[k].eq(a[k+'_old']).all() for k in ['fga','fgm','fg_points']))
a=reg.groupby(['athlete_id_1','region']).agg(attempts=('made','size'),makes=('made','sum'),field_goal_points=('points','sum')).reset_index()
b=pd.read_csv(REFERENCE/'player_regions.csv');a=b.merge(a,on=['athlete_id_1','region'],how='outer',suffixes=['_old','']).fillna(0)
checks['every_player_region_matches']=all(a[k].eq(a[k+'_old']).all() for k in ['attempts','makes','field_goal_points'])
assert checks['game_team_totals_match'] and checks['every_player_region_matches']
assert len(reg)==38535 and reg.made.sum()==16926 and reg.points.sum()==38533
assert s.points_attempted.isin([2,3]).all() and s.points.eq(s.made*s.points_attempted).all()
assert np.allclose(s.x,np.round(s.x)) and np.allclose(s.y,np.round(s.y))
assert not s[['x','y','made','date','game_id','player']].isna().any().any()
assert s.y.between(-25,25).all() and s.x.between(-5.25,88.75).all()
s['three_attempt']=s.points_attempted.eq(3).astype(int);s['three_made']=s.three_attempt*s.made
agg=s.groupby(['game_id','team_id']).agg(fga=('made','size'),fgm=('made','sum'),three_attempts=('three_attempt','sum'),three_makes=('three_made','sum')).reset_index()
box=pd.read_parquet(DATA/'team_box_2025.parquet')
b=agg.merge(box[['game_id','team_id','field_goals_attempted','field_goals_made','three_point_field_goals_attempted','three_point_field_goals_made']],on=['game_id','team_id'],validate='one_to_one',how='outer')
for left,right in [('fga','field_goals_attempted'),('fgm','field_goals_made'),('three_attempts','three_point_field_goals_attempted'),('three_makes','three_point_field_goals_made')]:assert len(b)==624 and b[left].eq(b[right]).all()
b.to_csv(RESULTS/'attempt_value_reconciliation.csv',index=False)
checks['attempt_values_match_all_624_team_boxes']=True
conflicts=s[~np.isclose(s.home_score-s.before_home_score+s.away_score-s.before_away_score,s.points)]
conflicts[['game_id','game_play_number','scope','text','points','home_score','away_score','before_home_score','before_away_score']].to_csv(RESULTS/'score_sequence_conflicts.csv',index=False)
checks.update({'regular_rows':len(reg),'regular_games':int(reg.game_id.nunique()),'regular_makes':int(reg.made.sum()),'regular_points':int(reg.points.sum()),'missing_players':int(reg.player.isna().sum()),'source_rows':len(d),'score_sequence_conflicts':len(conflicts),'regular_games_with_score_sequence_conflicts':int(conflicts[conflicts.scope.eq('regular')].game_id.nunique()),'sources':{name:{'url':url,'sha256':sha} for name,(url,sha) in sources.items()},'retrieved':'2026-11-13','comparison_note':'Public release matches saved game team totals and all player region counts; original row level archive is offline, so byte identity is not asserted.','regular_attempt_value_counts':reg.points_attempted.value_counts().astype(int).to_dict()})
checks['geometry_value_conflicts_over_one_foot']=int((reg.points_attempted.eq(3).ne(reg.arc_offset_exact.gt(0)) & reg.arc_offset_exact.abs().gt(1)).sum())
assert checks['geometry_value_conflicts_over_one_foot']==0
for label,mask in [('train',reg.date.lt('2025-08-10')),('validation',reg.date.ge('2025-08-10')&reg.date.lt('2025-08-27')),('test',reg.date.ge('2025-08-27'))]:checks[label+'_half_court']=int((mask&reg.x.between(-5.25,41.75)).sum())
(RESULTS/'source_audit.json').write_text(json.dumps(checks,indent=2))
s.to_csv(DATA/'shots.csv.gz',index=False)
print(json.dumps(checks,indent=2))
