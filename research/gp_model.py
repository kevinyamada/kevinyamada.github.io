import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[name]='1'
os.environ['PYTENSOR_FLAGS']='cxx='
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy.special import expit
from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from court_geometry import court
from paths import HERE, DATA, RESULTS, REFERENCE

ROOT=HERE
OUT=RESULTS/'gp'
OUT.mkdir(exist_ok=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--players',action='store_true')
    parser.add_argument('--basis',type=int,default=16)
    parser.add_argument('--draws',type=int,default=1000)
    parser.add_argument('--tune',type=int,default=1000)
    args=parser.parse_args()
    label=('player' if args.players else 'location')+str(args.basis)
    dest=OUT/label
    dest.mkdir(exist_ok=True)
    all_shots=pd.read_csv(DATA/'shots.csv.gz')
    s=all_shots[all_shots.scope.isin(['regular','playoffs']) & all_shots.x.between(-5.25,41.75)].copy()
    s['loc']=((s.x+5)*51+(s.y+25)).astype(int)
    grid=np.array([(x,y) for x in range(-5,42) for y in range(-25,26)],dtype=float)
    train=s[s.scope.eq('regular') & s.date.lt('2025-08-10')]
    ids=sorted(train.athlete_id_1.unique())
    player_index={v:i for i,v in enumerate(ids)}
    group=['loc','athlete_id_1'] if args.players else ['loc']
    a=train.groupby(group).made.agg(['sum','count']).reset_index()
    started=time.time()
    with pm.Model() as model:
        intercept=pm.Normal('intercept',mu=0,sigma=1)
        ell=pm.TruncatedNormal('ell',mu=8,sigma=3,lower=3,upper=20,shape=2)
        eta=pm.HalfNormal('eta',sigma=1)
        cov=eta**2*pm.gp.cov.Matern52(2,ls=ell)
        gp=pm.gp.HSGP(m=[args.basis,args.basis],L=[36,38],drop_first=True,cov_func=cov)
        phi,sqrt_psd=gp.prior_linearized(grid)
        phi_value=phi.eval()
        z=pm.Normal('z',mu=0,sigma=1,shape=phi_value.shape[1])
        weights=pm.Deterministic('weights',z*sqrt_psd)
        spatial=pt.dot(phi_value,weights)
        linear=intercept+spatial[a['loc'].to_numpy()]
        if args.players:
            sigma_player=pm.HalfNormal('sigma_player',sigma=.5)
            z_player=pm.Normal('z_player',mu=0,sigma=1,shape=len(ids))
            alpha=pm.Deterministic('alpha',sigma_player*z_player)
            linear=linear+alpha[a.athlete_id_1.map(player_index).to_numpy()]
        pm.Binomial('made',n=a['count'].to_numpy(),logit_p=linear,observed=a['sum'].to_numpy())
        print('SAMPLING',label,'groups',len(a),'shots',len(train),flush=True)
        trace=pm.sample(draws=args.draws,tune=args.tune,chains=4,cores=4,target_accept=.97,nuts_sampler='nutpie',random_seed=2026,progressbar=False,compute_convergence_checks=True)
    variables=['intercept','ell','eta','z']+(['sigma_player','z_player'] if args.players else [])
    summary=az.summary(trace,var_names=variables,round_to=6)
    summary.to_csv(dest/'diagnostics.csv')
    post=trace['posterior'].to_dataset()
    stats=trace['sample_stats'].to_dataset()
    def draws(name):
        v=post[name]
        return v.stack(sample=('chain','draw')).transpose('sample',...).values
    weights=draws('weights')
    intercept_draws=draws('intercept')
    alpha_draws=draws('alpha') if args.players else None
    np.savez_compressed(dest/'posterior.npz',weights=weights,intercept=intercept_draws,phi=phi_value,grid=grid,alpha=alpha_draws if args.players else np.empty((0,0)),players=ids,ell=draws('ell'),eta=draws('eta'))
    diagnostic={'chains':4,'draws_per_chain':args.draws,'tune_per_chain':args.tune,'max_rhat':float(summary.r_hat.max()),'minimum_ess_bulk':float(summary.ess_bulk.min()),'minimum_ess_tail':float(summary.ess_tail.min()),'divergences':int(stats.diverging.sum()),'seconds':time.time()-started,'basis_per_axis':args.basis,'training_shots':len(train),'training_players':len(ids),'training_groups':len(a),'pymc_version':pm.__version__,'excluded_beyond_half_court':int((all_shots.scope.eq('regular') & all_shots.x.gt(41.75)).sum())}
    diagnostic['sampling_checks_pass']=bool(diagnostic['max_rhat']<1.01 and diagnostic['minimum_ess_bulk']>=400 and diagnostic['divergences']==0)
    (dest/'diagnostics.json').write_text(json.dumps(diagnostic,indent=2))
    rng=np.random.default_rng(2026)
    indices=np.linspace(0,len(weights)-1,min(2000,len(weights)),dtype=int)
    latent=intercept_draws[indices,None]+weights[indices]@phi_value.T
    base=expit(latent)
    surface=pd.DataFrame({'x':grid[:,0],'y':grid[:,1],'mean':base.mean(axis=0),'lower':np.quantile(base,.025,axis=0),'upper':np.quantile(base,.975,axis=0)})
    surface.to_csv(dest/'surface.csv',index=False)
    rows=[]
    predictions=[]
    for split,mask in [('validation',s.scope.eq('regular')&s.date.ge('2025-08-10')&s.date.lt('2025-08-27')),('test',s.scope.eq('regular')&s.date.ge('2025-08-27')),('playoffs',s.scope.eq('playoffs'))]:
        b=s[mask].copy()
        means=[]
        for start in range(0,len(b),500):
            block=b.iloc[start:start+500]
            lin=latent[:,block['loc'].to_numpy()].copy()
            if args.players:
                for j,p in enumerate(block.athlete_id_1):
                    if p in player_index:lin[:,j]+=alpha_draws[indices,player_index[p]]
                    else:lin[:,j]+=rng.normal(size=len(indices))*draws('sigma_player')[indices]
            means.extend(expit(lin).mean(axis=0))
        means=np.array(means)
        for name,probs in [(label,means),('constant',np.repeat(train.made.mean(),len(b)))]:
            rows.append({'split':split,'model':name,'shots':len(b),'log_loss':log_loss(b.made,probs),'brier':brier_score_loss(b.made,probs),'auc':roc_auc_score(b.made,probs),'mean_prediction':probs.mean(),'observed':b.made.mean()})
        b['prediction']=means
        b['split']=split
        predictions.append(b[['game_id','game_play_number','split','made','prediction']])
    pd.DataFrame(rows).to_csv(dest/'metrics.csv',index=False)
    pd.concat(predictions).to_csv(dest/'predictions.csv',index=False)
    if args.players:
        info=pd.read_csv(REFERENCE/'player_summary.csv').set_index('athlete_id_1')
        table=pd.DataFrame({'athlete_id':ids,'mean_log_odds_effect':alpha_draws.mean(axis=0),'lower':np.quantile(alpha_draws,.025,axis=0),'upper':np.quantile(alpha_draws,.975,axis=0)})
        table['name']=table.athlete_id.map(info.name)
        table['training_attempts']=table.athlete_id.map(train.groupby('athlete_id_1').size())
        table.to_csv(dest/'player_effects.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    regular=s[s.scope.eq('regular')]
    xe=np.r_[-5.25,np.arange(-4.5,41,1),41.75]
    ye=np.r_[-25,np.arange(-24.5,25,1),25]
    counts=np.histogram2d(regular.x,regular.y,bins=[xe,ye])[0]
    density=counts/(np.diff(xe)[:,None]*np.diff(ye)[None,:])
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('white')
    h=axes[0].pcolormesh(ye,xe,np.ma.masked_equal(density,0),cmap=cmap,norm=LogNorm(vmin=1,vmax=density.max()))
    fig.colorbar(h,ax=axes[0],shrink=.7,label='Attempts per square foot (log scale)')
    axes[0].set_title('Observed shot frequency\nAll regular season attempts')
    for ax,col,title in [(axes[1],'mean','Posterior make probability'),(axes[2],'width','Uncertainty interval width')]:
        values=(surface.upper-surface.lower).to_numpy() if col=='width' else surface[col].to_numpy()
        h=ax.pcolormesh(ye,xe,values.reshape(47,51),cmap='viridis',vmin=0,vmax=1)
        fig.colorbar(h,ax=ax,shrink=.7,label='Probability' if col=='mean' else '95% credible interval width')
        ax.set_title(title+'\nTraining data only; player effect = 0')
    for ax in axes:court(ax)
    fig.suptitle('WNBA spatial analysis | '+label+'\nA sparse shot frequency band does not imply a dip in make probability',fontsize=13)
    fig.savefig(dest/'spatial_graph.png',dpi=150)
    plt.close(fig)
    print('FINISHED',json.dumps(diagnostic),flush=True)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
