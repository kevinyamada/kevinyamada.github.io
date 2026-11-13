import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
import json,time,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score
from catboost import CatBoostClassifier
from scipy.spatial import cKDTree
from spatial_methods import IDW,Kriging,SPDE
from paths import DATA, RESULTS
OUT=RESULTS
s=pd.read_csv(DATA/'shots.csv.gz')
r=s[s.scope.eq('regular')&s.x.between(-5.25,41.75)].copy()
train=r[r.date.lt('2025-08-10')];val=r[r.date.ge('2025-08-10')&r.date.lt('2025-08-27')];test=r[r.date.ge('2025-08-27')]
assert not set(train.game_id)&set(test.game_id)
assert not set(train.game_id)&set(val.game_id)
assert train.date.max()<val.date.min()<test.date.min()
features=['x','y','distance','angle']
context=features+['period_number','seconds','home','athlete_id_1','team_id']
for a in [r,train,val,test]:
    for c in ['athlete_id_1','team_id']:a.loc[:,c]=a[c].astype(int)

def baseline(tr,te):
    edges=[0,4,8,12,16,20,22+1.75/12,24,28,100]
    g=tr.groupby(pd.cut(tr.distance,edges,right=False),observed=True).made.agg(['sum','count'])
    return pd.cut(te.distance,edges,right=False).map((g['sum']+1)/(g['count']+2)).astype(float).fillna(tr.made.mean()).to_numpy()

def score(y,p):
    assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    return {'brier':brier_score_loss(y,p),'log_loss':log_loss(y,p),'auc':roc_auc_score(y,p),'mean_prediction':float(np.mean(p)),'observed':float(np.mean(y))}

models={};selection=[]
for power in [1,2,3]:
    m=IDW(power=power).fit(train);loss=log_loss(val.made,m.predict(val));selection.append({'family':'IDW','setting':str(power),'validation_log_loss':loss});models['IDW '+str(power)]=m
for universal in [False,True]:
    name='Universal kriging' if universal else 'Ordinary kriging';m=Kriging(universal).fit(train);models[name]=m
    pd.DataFrame(m.variogram,columns=['distance','semivariance','pairs']).to_csv(OUT/(name.replace(' ','_')+'_variogram.csv'),index=False)
    print(name,'variogram',m.sill,m.scale,flush=True)
for rang in [4,8,16]:
    for sigma in [.3,.7,1.5]:
        start=time.time();m=SPDE(range_ft=rang,sigma=sigma,spacing=3).fit(train)
        loss=log_loss(val.made,m.predict(val));name=f'SPDE {rang} {sigma}';models[name]=m
        selection.append({'family':'SPDE','setting':name,'validation_log_loss':loss});print(name,loss,'seconds',round(time.time()-start,2),flush=True)
for ctx in [False,True]:
    cols=context if ctx else features;cats=['athlete_id_1','team_id'] if ctx else []
    for depth in [3,5]:
        m=CatBoostClassifier(iterations=600,depth=depth,learning_rate=.035,l2_leaf_reg=10,loss_function='Logloss',thread_count=2,random_seed=2026,verbose=False,allow_writing_files=False)
        m.fit(train[cols],train.made,cat_features=cats)
        name=('Context boosting' if ctx else 'Spatial boosting')+' '+str(depth)
        models[name]=(m,cols);loss=log_loss(val.made,m.predict_proba(val[cols])[:,1])
        selection.append({'family':'Context boosting' if ctx else 'Spatial boosting','setting':name,'validation_log_loss':loss});print(name,loss,flush=True)
selected={}
for family in ['IDW','SPDE','Context boosting','Spatial boosting']:
    item=min([a for a in selection if a['family']==family],key=lambda a:a['validation_log_loss'])
    key='IDW '+item['setting'] if family=='IDW' else item['setting'];selected[family]=models[key]
selected.update({k:models[k] for k in ['Universal kriging','Ordinary kriging']})
pd.DataFrame(selection).to_csv(OUT/'validation_selection.csv',index=False)
(OUT/'model_parameters.json').write_text(json.dumps({name:{'sill':float(selected[name].sill),'scale':float(selected[name].scale)} for name in ['Universal kriging','Ordinary kriging']},indent=2))
rows=[];pred=[]
for split,a in [('validation',val),('test',test),('playoffs',s[s.scope.eq('playoffs')&s.x.between(-5.25,41.75)])]:
    p={'Distance baseline':baseline(train,a)}
    for name,m in selected.items():p[name]=m[0].predict_proba(a[m[1]])[:,1] if isinstance(m,tuple) else m.predict(a)
    p['Ensemble']=(p['Distance baseline']+p['SPDE']+p['Context boosting'])/3
    for name,v in p.items():rows.append({'split':split,'model':name,'shots':len(a),**score(a.made,v)})
    q=a[['game_id','game_play_number','made']].copy();q['split']=split
    for k,v in p.items():q[k]=v
    pred.append(q)
pd.DataFrame(rows).to_csv(OUT/'temporal_metrics.csv',index=False);pd.concat(pred).to_csv(DATA/'predictions.csv.gz',index=False)
grid=pd.DataFrame([(x,y) for x in range(-5,42) for y in range(-25,26)],columns=['x','y'])
p,lo,hi=selected['SPDE'].predict(grid,True)
grid['mean']=p;grid['lower']=lo;grid['upper']=hi;grid.to_csv(OUT/'spde_surface.csv',index=False)
mesh=selected['SPDE'];(OUT/'mesh.json').write_text(json.dumps({'nodes':mesh.nodes.tolist(),'triangles':mesh.mesh.simplices.tolist(),'range_ft':mesh.range_ft,'sigma':mesh.sigma,'spacing':mesh.spacing,'gradient_max':mesh.gradient_max,'inference':'Conditional Gaussian Laplace approximation with validation selected range and sigma; not INLA and not hyperparameter integrated.'}))
fine=SPDE(mesh.range_ft,mesh.sigma,spacing=2).fit(train);fp=fine.predict(test);coarse=mesh.predict(test)
(OUT/'mesh_sensitivity.json').write_text(json.dumps({'coarse_spacing_ft':3,'fine_spacing_ft':2,'mean_absolute_prediction_change':float(abs(fp-coarse).mean()),'max_absolute_prediction_change':float(abs(fp-coarse).max()),'fine_test':score(test.made,fp)},indent=2))
print('Temporal comparison complete',flush=True)
cv=[];folds=[]
for buffer in [0,2,4]:
    for fold in range(4):
        block=(train.x>=16).astype(int)*2+(train.y>=0).astype(int)
        te=train[block.eq(fold)]
        tr=train[~block.eq(fold)]
        near=cKDTree(te[['x','y']].drop_duplicates()).query(tr[['x','y']],k=1)[0]
        tr=tr[near>buffer].copy()
        minimum=cKDTree(te[['x','y']].drop_duplicates()).query(tr[['x','y']],k=1)[0].min()
        assert minimum>buffer and not set(tr.index)&set(te.index)
        folds.append({'buffer_ft':buffer,'fold':fold,'train_shots':len(tr),'test_shots':len(te),'minimum_separation_ft':minimum})
        preds={'Distance baseline':baseline(tr,te)}
        for name,m in [('IDW',IDW(2)),('Universal kriging',Kriging(True)),('Ordinary kriging',Kriging(False)),('SPDE',SPDE(8,.7,3))]:
            preds[name]=m.fit(tr).predict(te)
        boost=CatBoostClassifier(iterations=600,depth=3,learning_rate=.035,l2_leaf_reg=10,thread_count=2,random_seed=2026,verbose=False,allow_writing_files=False)
        boost.fit(tr[features],tr.made);preds['Spatial boosting']=boost.predict_proba(te[features])[:,1]
        preds['Spatial ensemble']=(preds['Distance baseline']+preds['SPDE']+preds['Spatial boosting'])/3
        for name,p in preds.items():cv.append({'buffer_ft':buffer,'fold':fold,'model':name,'shots':len(te),**score(te.made,p)})
        print('Spatial fold',buffer,fold,'done',flush=True)
pd.DataFrame(cv).to_csv(OUT/'spatial_cv.csv',index=False);pd.DataFrame(folds).to_csv(OUT/'spatial_folds.csv',index=False)
q=pd.concat(pred);a=q[q.split.eq('test')];rng=np.random.default_rng(2026);boot=[]
for name in list(selected)+['Ensemble']:
    e=(a[name]-a.made)**2-(a['Distance baseline']-a.made)**2
    g=pd.DataFrame({'game':a.game_id,'e':e}).groupby('game').e.agg(['sum','count'])
    w=rng.multinomial(len(g),np.repeat(1/len(g),len(g)),size=2000)
    values=(w@g['sum'])/(w@g['count'])
    boot.append({'model':name,'difference':e.mean(),'lower':np.quantile(values,.025),'upper':np.quantile(values,.975)})
pd.DataFrame(boot).to_csv(OUT/'game_bootstrap.csv',index=False)
print('FINISHED',flush=True)
