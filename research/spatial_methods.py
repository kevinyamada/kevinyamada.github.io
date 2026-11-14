import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='2'
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay,cKDTree,distance
from scipy.special import expit,logit
from scipy.optimize import least_squares
from scipy.linalg import cho_factor,cho_solve
from scipy.sparse import coo_matrix,diags,hstack,block_diag,csc_matrix
from scipy.sparse.linalg import splu


def aggregate(s,cell=2):
    a=s.assign(cx=np.floor(s.x/cell)*cell+cell/2,cy=np.floor(s.y/cell)*cell+cell/2)
    a=a.groupby(['cx','cy']).made.agg(['sum','count']).reset_index()
    a['rate']=(a['sum']+.5)/(a['count']+1)
    a['logit']=np.log((a['sum']+.5)/(a['count']-a['sum']+.5))
    a['variance']=1/(a['sum']+.5)+1/(a['count']-a['sum']+.5)
    return a

class IDW:
    def __init__(self,power=2,neighbors=16):self.power,self.neighbors=power,neighbors
    def fit(self,s):
        self.a=aggregate(s);self.tree=cKDTree(self.a[['cx','cy']]);return self
    def predict(self,s):
        d,j=self.tree.query(s[['x','y']],k=min(self.neighbors,len(self.a)))
        if d.ndim==1:d,j=d[:,None],j[:,None]
        w=np.maximum(d,.1)**(-self.power)
        exact=d[:,0]<1e-10
        p=(w*self.a.rate.to_numpy()[j]).sum(axis=1)/w.sum(axis=1)
        p[exact]=self.a.rate.to_numpy()[j[exact,0]]
        return p

class Kriging:
    def __init__(self,universal=True):self.universal=universal
    def trend(self,xy):
        r=np.hypot(xy[:,0],xy[:,1])/25
        return np.column_stack([np.ones(len(xy)),r,r*r]) if self.universal else np.ones((len(xy),1))
    def fit(self,s):
        self.a=aggregate(s);self.xy=self.a[['cx','cy']].to_numpy();self.F=self.trend(self.xy)
        z=self.a.logit.to_numpy();noise=self.a.variance.to_numpy()
        b=np.linalg.lstsq(self.F/np.sqrt(noise[:,None]),z/np.sqrt(noise),rcond=None)[0]
        residual=z-self.F@b
        i,j=np.triu_indices(len(z),1)
        d=np.linalg.norm(self.xy[i]-self.xy[j],axis=1)
        gamma=.5*((residual[i]-residual[j])**2-noise[i]-noise[j])
        rows=[]
        for lo in np.arange(0,32,2):
            m=(d>=lo)&(d<lo+2)&(self.a['count'].to_numpy()[i]>=20)&(self.a['count'].to_numpy()[j]>=20)
            if m.sum()>30:rows.append([d[m].mean(),gamma[m].mean(),int(m.sum())])
        if len(rows)<3:raise ValueError("Too few populated cells for a residual variogram")
        self.variogram=np.array(rows)
        def error(v):
            sill,scale=v
            return (sill*(1-np.exp(-self.variogram[:,0]/scale))-self.variogram[:,1])*np.sqrt(self.variogram[:,2]/self.variogram[:,2].max())
        fit=least_squares(error,[.3,8],bounds=([.001,1],[5,40]))
        self.sill,self.scale=fit.x
        self.K=self.sill*np.exp(-distance.cdist(self.xy,self.xy)/self.scale)+np.diag(noise+1e-8)
        self.chol=cho_factor(self.K,lower=True)
        self.kiF=cho_solve(self.chol,self.F)
        self.G=np.linalg.inv(self.F.T@self.kiF)
        self.beta=self.G@self.F.T@cho_solve(self.chol,z)
        self.alpha=cho_solve(self.chol,z-self.F@self.beta)
        return self
    def predict(self,s):
        xy=s[['x','y']].to_numpy();k=self.sill*np.exp(-distance.cdist(xy,self.xy)/self.scale)
        return expit(self.trend(xy)@self.beta+k@self.alpha)

class SPDE:
    def __init__(self,range_ft=8,sigma=.7,spacing=3):
        self.range_ft,self.sigma,self.spacing=range_ft,sigma,spacing
        self.nodes=np.array([(x,y) for x in np.arange(-23,64,spacing) for y in np.arange(-43,44,spacing)])
        self.mesh=Delaunay(self.nodes)
        n=len(self.nodes);mass=np.zeros(n);rows=[];cols=[];values=[]
        for tri in self.mesh.simplices:
            v=self.nodes[tri];M=np.column_stack([np.ones(3),v]);area=abs(np.linalg.det(M))/2
            grad=np.linalg.inv(M)[1:,:]
            g=area*grad.T@grad
            mass[tri]+=area/3
            for i in range(3):
                for j in range(3):rows.append(tri[i]);cols.append(tri[j]);values.append(g[i,j])
        C=diags(mass);G=coo_matrix((values,(rows,cols)),shape=(n,n)).tocsc()
        kappa=np.sqrt(8)/range_ft;tau=1/(np.sqrt(4*np.pi)*kappa*sigma)
        q=tau*tau*(kappa**4*C+2*kappa*kappa*G+G@diags(1/mass)@G)
        self.Q=block_diag([csc_matrix([[.25]]),q],format='csc')
        self.mass=mass
    def design(self,xy):
        xy=np.asarray(xy);t=self.mesh.find_simplex(xy)
        assert (t>=0).all()
        b=np.einsum('nij,nj->ni',self.mesh.transform[t,:2],xy-self.mesh.transform[t,2])
        b=np.column_stack([b,1-b.sum(axis=1)])
        assert np.allclose(b.sum(axis=1),1) and b.min()>-1e-8
        A=coo_matrix((b.ravel(),(np.repeat(np.arange(len(xy)),3),self.mesh.simplices[t].ravel())),shape=(len(xy),len(self.nodes)))
        return hstack([np.ones((len(xy),1)),A],format='csc')
    def fit(self,s):
        a=s.groupby(['x','y']).made.agg(['sum','count']).reset_index()
        B=self.design(a[['x','y']]);n=a['count'].to_numpy();k=a['sum'].to_numpy()
        theta=np.zeros(B.shape[1]);theta[0]=logit((k.sum()+1)/(n.sum()+2))
        def objective(t):
            e=B@t
            return float(np.sum(n*np.logaddexp(0,e)-k*e)+.5*t@(self.Q@t))
        self.converged=False
        for step in range(60):
            eta=B@theta;p=expit(eta);w=n*p*(1-p)
            grad=B.T@(n*p-k)+self.Q@theta
            H=(self.Q+B.T@diags(w)@B).tocsc()
            change=splu(H).solve(np.asarray(grad))
            scale=1.;old=objective(theta)
            while objective(theta-scale*change)>old+1e-9 and scale>1e-6:scale/=2
            theta-=scale*change
            if abs(scale*change).max()<1e-6:self.converged=True;break
        assert self.converged,'SPDE mode did not converge'
        p=expit(B@theta);self.H=(self.Q+B.T@diags(n*p*(1-p))@B).tocsc()
        self.factor=splu(self.H);self.theta=theta
        self.gradient_max=float(abs(B.T@(n*p-k)+self.Q@theta).max())
        assert self.gradient_max<.001
        return self
    def predict(self,s,uncertainty=False):
        xy=s[['x','y']].to_numpy();unique,inverse=np.unique(xy,axis=0,return_inverse=True)
        B=self.design(unique);mu=np.asarray(B@self.theta);var=np.zeros(len(mu))
        for start in range(0,len(mu),200):
            a=B[start:start+200].toarray();sol=self.factor.solve(a.T)
            var[start:start+200]=np.einsum('ij,ji->i',a,sol)
        assert var.min()>-1e-8
        sd=np.sqrt(np.maximum(var,0));nodes,weights=np.polynomial.hermite.hermgauss(15)
        p=(expit(mu[:,None]+np.sqrt(2)*sd[:,None]*nodes)@weights)/np.sqrt(np.pi)
        if uncertainty:return p[inverse],expit(mu[inverse]-1.96*sd[inverse]),expit(mu[inverse]+1.96*sd[inverse])
        return p[inverse]
