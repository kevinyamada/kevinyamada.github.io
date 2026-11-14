import json
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from scipy.special import expit
from spatial_methods import SPDE,IDW,Kriging
rng=np.random.default_rng(2026)
s=pd.DataFrame({'x':rng.integers(0,16,8000),'y':rng.integers(-8,8,8000)})
s['made']=rng.binomial(1,expit(1-np.hypot(s.x,s.y)/15))
m=SPDE(8,.7,4).fit(s)
Q=m.Q.toarray();assert np.allclose(Q,Q.T) and np.linalg.eigvalsh(Q).min()>0
B=m.design(s[['x','y']]);assert np.allclose(np.asarray(B[:,1:].sum(axis=1)).ravel(),1)
p,lo,hi=m.predict(s,True);assert (lo<hi).all() and (p>0).all() and (p<1).all()
mean=np.asarray(B@m.theta)
H=m.H.toarray();dense=np.linalg.inv(H)
small=B[:5].toarray();expected=np.einsum('ij,jk,ik->i',small,dense,small)
actual=np.einsum('ij,ji->i',small,m.factor.solve(small.T))
assert np.allclose(actual,expected)
for model in [IDW(),Kriging(True),Kriging(False)]:
 model.fit(s);assert np.isfinite(model.predict(s)).all()
print(json.dumps({'precision_symmetric_positive_definite':True,'barycentric_weights_sum_to_one':True,'sparse_variances_match_dense_inverse':True,'mode_gradient_max':m.gradient_max,'probability_bounds':True}))
