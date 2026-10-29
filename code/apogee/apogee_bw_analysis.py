#!/usr/bin/env python3
"""APOGEE: covariance-calibrated abundance representations using the BW map.

Run from any directory: python apogee_bw_analysis.py
All results are recomputed from data/catalog_arrays.npz. If that file is absent,
the pinned public arrays are downloaded, hash checked, and read with a restricted
NumPy-only unpickler. No commercial solver or npeb installation is required.

The EB comparator is a finite smooth Gaussian location mixture, fitted by EM.
It is NOT the unrestricted NPMLE and is NOT the authors' original npeb code.
All covariance matrices use denominator n-1. Reported precisions are inverse
VARIANCES (not inverse standard deviations).
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import pickle
import time
import urllib.request
from pathlib import Path

# Set these before NumPy import for predictable CPU usage and numerical timing.
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import kmeans_plusplus
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
COMMIT = '71f03d26301f84bda307c27fdc3329b9a9a814ca'
BASE = f'https://raw.githubusercontent.com/aqjaffe/constrained-denoising-EB-OT/{COMMIT}/'
HASHES = {'X_chem.pkl': 'ad82fd4fbb6a4655ea5fa13765632a5ac1ad3929230f9d11c60120ad819a60e1',
          'chem_prec.pkl': '84e1240445300fd77becd5d65d5aa55e93db1754ea0a4738d484c8e19c6f32da'}
COLUMNS = ['MG_FE','FE_H','C_FE','CI_FE','N_FE','O_FE','AL_FE','CA_FE','TI_FE',
           'SI_FE','P_FE','S_FE','K_FE','TIII_FE','V_FE','CR_FE','MN_FE','CO_FE','NI_FE']
ELEMENTS = ['C','N','O','MG','AL','SI','S','K','CA','NI']
SETS = {'8': ['C','N','O','MG','AL','SI','CA','NI'],
        '10': ELEMENTS,
        '12': ['C','N','O','MG','AL','SI','S','K','CA','TI','MN','NI'],
        '15': ['C','N','O','MG','AL','SI','S','K','CA','TI','V','CR','MN','CO','NI']}
SEED = 20261021
FLOOR = 1e-7  # dex^2; only numerical/diagnostic safeguards, always recorded
METHODS = ['Observed', 'Direct BW', 'Smooth EB', 'EB + BW', 'Diagonal match']
COLORS = {'Observed':'#85898e', 'Direct BW':'#006f8b', 'Smooth EB':'#d97732',
          'EB + BW':'#7653a1', 'Diagonal match':'#758a47', 'Moment target':'#172a3a'}

def sym(a):
    return (a + a.T) / 2

def cov(x):
    z = x - x.mean(axis=0)
    return sym(z.T @ z / (len(x)-1))

def psd(a, floor=FLOOR):
    lam, u = np.linalg.eigh(sym(a))
    return sym((u * np.maximum(lam, floor)) @ u.T), int(np.sum(lam < floor))

def mpow(a, power):
    lam, u = np.linalg.eigh(sym(a))
    if lam.min() <= 0:
        raise ValueError(f'Matrix power requires SPD: min eigenvalue={lam.min()}')
    return sym((u * lam**power) @ u.T)

def bw_map(b, c):
    """Unique SPD M with M B M^T = C; do not replace by sqrt(C) invsqrt(B)."""
    h, ih = mpow(b, .5), mpow(b, -.5)
    return sym(ih @ mpow(sym(h @ c @ h), .5) @ ih)

def moments(y, v):
    b = cov(y)
    g = np.diag(v.mean(axis=0))
    raw = sym(b-g)
    c, nclip = psd(raw)
    return {'mu':y.mean(0), 'b':b, 'g':g, 'raw':raw, 'c':c, 'nclip':nclip}

def spectrum(a):
    return np.linalg.eigvalsh(sym(a))[::-1]

def spectral_summary(a):
    lam = np.maximum(spectrum(a), 0)
    prop = lam/lam.sum()
    return {'trace':float(lam.sum()), 'd90':int(np.searchsorted(np.cumsum(prop), .90)+1),
            'd95':int(np.searchsorted(np.cumsum(prop), .95)+1),
            'participation_ratio':float(1 / np.sum(prop**2))}

def commutator(a, b):
    return float(np.linalg.norm(a@b-b@a, 'fro') /
                 (np.linalg.norm(a,'fro')*np.linalg.norm(b,'fro')))

class RestrictedArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        import numpy.core.multiarray as ma
        allowed = {('numpy.core.multiarray','_reconstruct'):ma._reconstruct,
                   ('numpy._core.multiarray','_reconstruct'):ma._reconstruct,
                   ('numpy','ndarray'):np.ndarray, ('numpy','dtype'):np.dtype}
        if (module,name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported global: {module}.{name}')
        return allowed[(module,name)]

def get_data():
    path = ROOT/'data'/'catalog_arrays.npz'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        arrays=[]
        for name, expected in HASHES.items():
            data=urllib.request.urlopen(BASE+'data/'+name, timeout=90).read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f'Checksum mismatch: {name}')
            arrays.append(RestrictedArrayUnpickler(io.BytesIO(data)).load())
        np.savez_compressed(path, abundance=arrays[0], precision=arrays[1], columns=COLUMNS)
    f=np.load(path, allow_pickle=False)
    if list(f['columns']) != COLUMNS:
        raise ValueError('Unexpected column order')
    return f['abundance'], 1/f['precision']

def select_columns(y, v, elems):
    j=[COLUMNS.index(e+'_FE') for e in elems]
    yy, vv=y[:,j], v[:,j]
    ok=np.isfinite(yy).all(1)&np.isfinite(vv).all(1)&(vv>0).all(1)
    return yy[ok],vv[ok],np.flatnonzero(ok)

class SmoothEB:
    """Finite mixture G=sum_k w_k N(a_k,tau^2 I), with heteroskedastic noise.

    EM updates the weights and locations; component covariance is fixed.
    This is a local likelihood fit over a finite sieve, not a certified NPMLE.
    Initialization uses kmeans++ on the noisy training observations only.
    """
    def __init__(self, tau=.025, k=64, max_iter=1200, tol=1e-6, seed=SEED):
        self.tau,self.k,self.max_iter,self.tol,self.seed=tau,k,max_iter,tol,seed

    def _kernel(self,y,v):
        iv=1/(v+self.tau**2)
        return ((y*iv)@self.atoms.T - .5*iv@(self.atoms**2).T
                - .5*np.sum(y*y*iv+np.log(2*np.pi*(v+self.tau**2)),axis=1)[:,None])

    def fit(self,y,v):
        start=time.perf_counter()
        self.atoms=kmeans_plusplus(y,n_clusters=self.k,random_state=self.seed)[0]
        self.weights=np.full(self.k,1/self.k)
        iv=1/(v+self.tau**2)
        yiv=y*iv
        const=-.5*np.sum(y*y*iv+np.log(2*np.pi*(v+self.tau**2)),axis=1)
        previous=-np.inf
        self.converged=False
        self.trace=[]
        for it in range(self.max_iter):
            logp=yiv@self.atoms.T-.5*iv@(self.atoms**2).T+const[:,None]
            logp+=np.log(np.maximum(self.weights,1e-300))[None,:]
            norm=logsumexp(logp,axis=1)
            ll=float(norm.mean())
            self.trace.append(ll)
            if np.isfinite(previous) and ll < previous-1e-8:
                raise RuntimeError('EM likelihood decreased')
            if np.isfinite(previous) and ll-previous < self.tol:
                self.converged=True
                break
            r=np.exp(logp-norm[:,None])
            self.weights=np.maximum(r.mean(0),1e-300)
            self.weights/=self.weights.sum()
            self.atoms=(r.T@yiv)/np.maximum(r.T@iv,1e-300)
            previous=ll
        self.n_iter=it+1
        self.loglik=self.score(y,v)
        self.seconds=time.perf_counter()-start
        return self

    def score(self,y,v):
        return float(logsumexp(self._kernel(y,v)+np.log(self.weights)[None,:],axis=1).mean())

    def posterior(self,y,v):
        logp=self._kernel(y,v)+np.log(self.weights)[None,:]
        r=np.exp(logp-logsumexp(logp,axis=1)[:,None])
        a=r@self.atoms
        return (self.tau**2*y+v*a)/(self.tau**2+v)

    def diagnostics(self):
        return {'tau':self.tau,'components':self.k,'iterations':self.n_iter,
                'converged':self.converged,'train_loglik':self.loglik,'fit_seconds':self.seconds,
                'last_increment':self.trace[-1]-self.trace[-2] if len(self.trace)>1 else None}

def tune_eb(y,v,train,val,k,max_iter,seed):
    rows=[]
    for tau in [.01,.025,.05]:
        model=SmoothEB(tau,k,max_iter,seed=seed).fit(y[train],v[train])
        diag=model.diagnostics()
        diag['validation_loglik']=model.score(y[val],v[val])
        rows.append(diag)
    best=max(rows,key=lambda z:z['validation_loglik'])['tau']
    return best,rows

def fit_maps(y,v,model):
    t0=time.perf_counter()
    m=moments(y,v)
    m['bw']=bw_map(m['b'],m['c'])
    m['direct_seconds']=time.perf_counter()-t0
    m['diag']=np.diag(np.sqrt(np.diag(m['c'])/np.diag(m['b'])))
    eb=model.posterior(y,v)
    eb_cov,nclip=psd(cov(eb))
    m['eb_mu']=eb.mean(0)
    m['eb_cov']=eb_cov
    m['eb_source_clips']=nclip
    m['eb_bw']=bw_map(eb_cov,m['c'])
    return m

def predict(y,v,model,m):
    eb=model.posterior(y,v)
    return {'Observed':y,
            'Direct BW':m['mu']+(y-m['mu'])@m['bw'].T,
            'Smooth EB':eb,
            'EB + BW':m['mu']+(eb-m['eb_mu'])@m['eb_bw'].T,
            'Diagonal match':m['mu']+(y-m['mu'])@m['diag'].T}

def scores(z,target,reference=None):
    s=cov(z)
    out={'covariance_error':float(np.linalg.norm(s-target,'fro')/np.linalg.norm(target,'fro')),
         'spectral_error':float(np.linalg.norm(spectrum(s)-spectrum(target))/np.linalg.norm(spectrum(target))),
         **spectral_summary(s)}
    if reference is not None:
        out['rmse']=float(np.sqrt(np.mean((z-reference)**2)))
    return out

def write_json(path,obj):
    def default(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj,indent=2,default=default)+'\n')

def run(args):
    for p in ['results','figures']:(ROOT/p).mkdir(exist_ok=True)
    raw,raw_v=get_data()
    y,v,indices=select_columns(raw,raw_v,ELEMENTS)
    n,d=y.shape
    rng=np.random.default_rng(SEED)
    order=rng.permutation(n)
    nt,nv=int(.6*n),int(.2*n)
    tr,va,te=order[:nt],order[nt:nt+nv],order[nt+nv:]
    np.savez_compressed(ROOT/'results'/'split_indices.npz',train=tr,validation=va,test=te,
                        retained_catalog_rows=indices)
    print(f'Catalog: n={n}, d={d}; train/validation/test={len(tr)}/{len(va)}/{len(te)}',flush=True)
    tau,tuning=tune_eb(y,v,tr,va,args.components,args.max_iter,SEED)
    print(f'EB bandwidth selected by validation likelihood: tau={tau}',flush=True)
    pd.DataFrame(tuning).to_csv(ROOT/'results'/'eb_tuning.csv',index=False)
    fitidx=np.r_[tr,va]
    held_model=SmoothEB(tau,args.components,args.max_iter,seed=SEED).fit(y[fitidx],v[fitidx])
    held_maps=fit_maps(y[fitidx],v[fitidx],held_model)
    pred=predict(y[te],v[te],held_model,held_maps)
    test_m=moments(y[te],v[te])
    held_scores=[{'method':name,**scores(z,test_m['raw'])} for name,z in pred.items()]
    pd.DataFrame(held_scores).to_csv(ROOT/'results'/'heldout_scores.csv',index=False)
    print('Held-out scores:',held_scores,flush=True)
    # Bootstrap is conditional on the fixed fitted maps. Resample star records,
    # including observations, uncertainty profiles, and every method's output.
    held_boot=[]
    for b in range(args.bootstrap):
        ix=rng.integers(0,len(te),len(te))
        target=moments(y[te][ix],v[te][ix])['raw']
        for name,z in pred.items():
            held_boot.append({'replicate':b,'method':name,**scores(z[ix],target)})
    pd.DataFrame(held_boot).to_csv(ROOT/'results'/'heldout_bootstrap.csv',index=False)

    print('Fitting descriptive full-catalog representation...',flush=True)
    full_model=SmoothEB(tau,args.components,args.max_iter,seed=SEED).fit(y,v)
    m=fit_maps(y,v,full_model)
    full=predict(y,v,full_model,m)
    allcov={name:cov(z) for name,z in full.items()}
    allcov['Moment target']=m['c']
    descript=[{'method':name,**scores(z,m['c'])} for name,z in full.items()]
    pd.DataFrame(descript).to_csv(ROOT/'results'/'catalog_scores.csv',index=False)
    eig_rows=[]
    for name,c in allcov.items():
        eig=spectrum(c)
        for j,l in enumerate(eig):
            eig_rows.append({'method':name,'component':j+1,'eigenvalue':l,
                             'cumulative_fraction':np.sum(eig[:j+1])/np.sum(eig)})
    pd.DataFrame(eig_rows).to_csv(ROOT/'results'/'catalog_spectra.csv',index=False)
    _,vectors=np.linalg.eigh(m['c'])
    vectors=vectors[:,::-1]
    for j in range(d):
        if vectors[np.argmax(np.abs(vectors[:,j])),j]<0:
            vectors[:,j]*=-1
    pd.DataFrame(vectors,index=ELEMENTS,columns=[f'PC{j+1}' for j in range(d)]).to_csv(ROOT/'results'/'pc_loadings.csv')

    boots=[]
    for b in range(args.bootstrap):
        ix=rng.integers(0,n,n)
        bm=moments(y[ix],v[ix])
        eig=spectrum(bm['raw'])
        boots.append(eig)
    boots=np.asarray(boots)
    np.save(ROOT/'results'/'target_spectrum_bootstrap.npy',boots)
    bsum=pd.DataFrame({'component':np.arange(1,d+1),'estimate':spectrum(m['raw']),
                      'lower_025':np.quantile(boots,.025,axis=0),
                      'upper_975':np.quantile(boots,.975,axis=0)})
    bsum.to_csv(ROOT/'results'/'spectrum_intervals.csv',index=False)

    # Sensitivity analyses are descriptive, not post-selection hypothesis tests.
    sens=[]
    for label,elems in SETS.items():
        sy,sv,_=select_columns(raw,raw_v,elems)
        sm=moments(sy,sv)
        for scale in [.5,.75,1,1.25,1.5]:
            c=sm['b']-scale*sm['g']
            p,nclip=psd(c)
            sens.append({'dimension':int(label),'variance_multiplier':scale,
                         'error_correlation':0.,'n':len(sy),'min_raw_eigenvalue':spectrum(c)[-1],
                         'negative_eigenvalues':int(np.sum(spectrum(c)<0)),
                         'floor_count':nclip,**spectral_summary(p)})
    for rho in [.1,.25]:
        sd=np.sqrt(v)
        g=(1-rho)*np.diag(v.mean(0))+rho*(sd.T@sd/n)
        c=m['b']-g
        p,nclip=psd(c)
        sens.append({'dimension':d,'variance_multiplier':1.,'error_correlation':rho,'n':n,
                     'min_raw_eigenvalue':spectrum(c)[-1],
                     'negative_eigenvalues':int(np.sum(spectrum(c)<0)),
                     'floor_count':nclip,**spectral_summary(p)})
    pd.DataFrame(sens).to_csv(ROOT/'results'/'sensitivity.csv',index=False)

    # An arbitrary covariance-matching whitening/recoloring map need not be SPD.
    w=mpow(m['c'],.5)@mpow(m['b'],-.5)
    wc=m['mu']+(y-m['mu'])@w.T
    whiten={'relative_asymmetry':float(np.linalg.norm(w-w.T,'fro')/np.linalg.norm(w,'fro')),
            'bw_displacement':float(np.mean(np.sum((full['Direct BW']-y)**2,axis=1))),
            'whitening_displacement':float(np.mean(np.sum((wc-y)**2,axis=1))),
            'whitening_cov_error':scores(wc,m['c'])['covariance_error']}

    # Data-based extra-noise benchmark: observed catalog vectors are reference
    # signals, NOT astrophysical ground truth. Uncertainty profiles are sampled
    # independently of the reference row. Test reference vectors are used only
    # for scoring. Every replicate tunes EB on an internal validation partition.
    injection=[];inj_fits=[]
    for rep in range(args.replicates):
        irng=np.random.default_rng(SEED+1000+rep)
        take=irng.choice(n,min(args.benchmark_n,n),replace=False)
        ref=y[take].copy()
        vv=v[irng.integers(0,n,len(ref))].copy()
        yy=ref+irng.normal(size=ref.shape)*np.sqrt(vv)
        ii=irng.permutation(len(ref));na,nb=int(.6*len(ref)),int(.2*len(ref))
        ia,ib,ic=ii[:na],ii[na:na+nb],ii[na+nb:]
        btau,tune=tune_eb(yy,vv,ia,ib,args.components,args.max_iter,SEED+rep)
        fit=np.r_[ia,ib]
        model=SmoothEB(btau,args.components,args.max_iter,seed=SEED+rep).fit(yy[fit],vv[fit])
        maps=fit_maps(yy[fit],vv[fit],model)
        out=predict(yy[ic],vv[ic],model,maps)
        target=cov(ref[ic])
        for name,z in out.items():
            injection.append({'replicate':rep,'method':name,**scores(z,target,ref[ic])})
        inj_fits.append({'replicate':rep,'tuning':tune,'fit':model.diagnostics(),
                         'target_floors':maps['nclip']})
        print(f'Extra-noise replicate {rep+1}/{args.replicates} finished',flush=True)
    idf=pd.DataFrame(injection)
    idf.to_csv(ROOT/'results'/'extra_noise_replicates.csv',index=False)
    isum=idf.groupby('method',sort=False)[['covariance_error','spectral_error','rmse']].agg(['mean','std'])
    isum.columns=['_'.join(x) for x in isum.columns]
    isum.reset_index().to_csv(ROOT/'results'/'extra_noise_summary.csv',index=False)

    diag={'seed':SEED,'catalog_n':n,'d':d,'elements':ELEMENTS,
          'main_set_choice':'Exploratory: a 12-coordinate trial had two negative corrected eigenvalues; omit Ti and Mn for the positive-definite main example. All sets and noise sensitivities are reported.',
          'source_commit':COMMIT,'source_sha256':HASHES,'arrays_sha256':hashlib.sha256((ROOT/'data'/'catalog_arrays.npz').read_bytes()).hexdigest(),
          'duplicate_full_rows':int(len(raw)-len(np.unique(raw,axis=0))),
          'train_n':len(tr),'validation_n':len(va),'test_n':len(te),
          'main_tuning':tuning,'heldout_eb':held_model.diagnostics(),'full_eb':full_model.diagnostics(),
          'full_target_floors':m['nclip'],'heldout_training_floors':held_maps['nclip'],
          'heldout_reference_floors':test_m['nclip'],'full_eb_source_floors':m['eb_source_clips'],
          'full_direct_seconds':m['direct_seconds'],
          'min_target_eigenvalue':spectrum(m['raw'])[-1],
          'noise_trace_fraction':np.trace(m['g'])/np.trace(m['b']),
          'commutator_B_Omega':commutator(m['b'],m['c']),
          'commutator_Gamma_Omega':commutator(m['g'],m['c']),
          'whitening':whiten,'target_summary':spectral_summary(m['c']),
          'bootstrap_d95_counts':{str(k):int(sum(spectral_summary(np.diag(e))['d95']==k for e in boots)) for k in range(1,d+1)},
          'benchmark_replicates':args.replicates,'benchmark_n':args.benchmark_n,
          'benchmark_fits':inj_fits,'args':vars(args)}
    write_json(ROOT/'results'/'diagnostics.json',diag)
    np.savez_compressed(ROOT/'results'/'fitted_arrays.npz',mean=m['mu'],observed_cov=m['b'],
                        mean_noise_cov=m['g'],target_cov=m['c'],raw_target_cov=m['raw'],
                        bw_map=m['bw'],eb_bw_map=m['eb_bw'],eb_cov=m['eb_cov'],
                        eb_atoms=full_model.atoms,eb_weights=full_model.weights,tau=tau,
                        **{name.replace(' ','_').replace('+','plus'):z for name,z in full.items()})
    # Checks are scientific identities, not tests that merely repeat the code.
    checks={'bw_cov_identity':float(np.linalg.norm(m['bw']@m['b']@m['bw'].T-m['c'],'fro')/np.linalg.norm(m['c'],'fro')),
            'bw_symmetry':float(np.linalg.norm(m['bw']-m['bw'].T)),
            'bw_min_eigenvalue':float(np.linalg.eigvalsh(m['bw']).min()),
            'eb_bw_cov_identity':scores(full['EB + BW'],m['c'])['covariance_error'],
            'bw_not_worse_displacement_than_whitening':whiten['bw_displacement']<=whiten['whitening_displacement']+1e-12}
    assert checks['bw_cov_identity']<1e-9
    assert checks['eb_bw_cov_identity']<1e-9
    assert checks['bw_min_eigenvalue']>0
    assert checks['bw_not_worse_displacement_than_whitening']
    write_json(ROOT/'results'/'numerical_checks.json',checks)
    make_figures(allcov,boots,m,idf,sens)
    print('Analysis complete.',flush=True)

def setup_plot():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.titlesize':15,
                         'axes.labelsize':13,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','savefig.facecolor':'white'})

def make_figures(allcov,boots,m,idf,sens):
    setup_plot()
    d=len(ELEMENTS);x=np.arange(1,d+1)
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.15),layout='constrained')
    lo,hi=np.quantile(boots,[.025,.975],axis=0)
    axes[0].fill_between(x,lo,hi,color=COLORS['Moment target'],alpha=.12,label='Target: 95% bootstrap')
    # The three calibrated spectra are numerically identical; one visible line
    # avoids redundant curves and lets the legend remain legible in the paper.
    for name in ['Observed','Smooth EB','Direct BW']:
        l=spectrum(allcov[name]);style='-'
        label='BW maps / moment target' if name=='Direct BW' else name
        axes[0].plot(x,l,style,color=COLORS[name],marker='o' if name not in ['EB + BW','Moment target'] else None,
                     ms=3,lw=1.6,label=label)
        axes[1].plot(x,np.cumsum(l)/l.sum(),style,color=COLORS[name],lw=1.7,label=label)
    axes[0].set(yscale='log',xlabel='Principal component',ylabel='Covariance eigenvalue (dex$^2$)',
                title='A  |  Population variance across directions',xticks=x)
    axes[0].legend(frameon=False,fontsize=11,loc='lower left')
    axes[1].axhline(.95,color='#a4a9ae',lw=1,ls=':')
    axes[1].text(1.12,.957,'95% of total variance',fontsize=11,color='#666666')
    axes[1].set(xlabel='Number of principal components',ylabel='Cumulative variance fraction',
                title='B  |  Global linear dimension summary',xticks=x,ylim=(.4,1.015))
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(ROOT/'figures'/'apogee_spectrum.png',dpi=240)
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12.2,4.1),layout='constrained')
    den=np.sqrt(np.diag(m['c']));corr=m['c']/den[:,None]/den[None,:]
    mats=[corr,m['bw']-np.eye(d),m['eb_bw']-np.eye(d)]
    titles=['A  |  Corrected latent correlation','B  |  Direct BW map minus identity','C  |  EB correction map minus identity']
    labels=[s.title() for s in ELEMENTS]
    for ax,a,title in zip(axes,mats,titles):
        lim=1 if ax is axes[0] else np.max(np.abs(a))
        im=ax.imshow(a,cmap='RdBu_r',vmin=-lim,vmax=lim)
        ax.set(xticks=np.arange(d),yticks=np.arange(d),xticklabels=labels,yticklabels=labels,title=title)
        ax.set_title(title,fontsize=12)
        ax.tick_params(axis='x',rotation=60,labelsize=12);ax.tick_params(axis='y',labelsize=12)
        fig.colorbar(im,ax=ax,shrink=.75,pad=.02)
    fig.savefig(ROOT/'figures'/'apogee_noncommuting.png',dpi=240)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11.6,4.15),layout='constrained')
    labels=['Observed','Direct\nBW','Smooth\nEB','EB +\nBW','Diagonal\nmatch']
    for ax,metric,title,ylabel in zip(axes,['spectral_error','rmse'],
            ['A  |  Reference spectrum recovery','B  |  Individual reference accuracy'],
            ['Relative eigenvalue error','RMSE per coordinate (dex)']):
        for j,name in enumerate(METHODS):
            values=idf.loc[idf.method==name,metric].values
            ax.bar(j,values.mean(),color=COLORS[name],alpha=.85,width=.65)
            ax.errorbar(j,values.mean(),yerr=values.std(ddof=1),color='#27343b',capsize=3,lw=1)
        ax.set(xticks=np.arange(len(labels)),xticklabels=labels,ylabel=ylabel,title=title)
        ax.grid(axis='y',alpha=.15)
    fig.savefig(ROOT/'figures'/'apogee_extra_noise_validation.png',dpi=240)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11.6,4.15),layout='constrained')
    ss=pd.DataFrame(sens);ss=ss[ss.error_correlation==0]
    for dim,color in zip([8,10,12,15],['#85898e','#006f8b','#d97732','#7653a1']):
        z=ss[ss.dimension==dim]
        axes[0].plot(z.variance_multiplier,z.min_raw_eigenvalue,'o-',color=color,label=f'{dim} elements')
        axes[1].plot(z.variance_multiplier,z.d95,'o-',color=color,label=f'{dim} elements')
    axes[0].axhline(0,color='#28343b',ls=':',lw=1)
    axes[0].set(title='A  |  Positivity before any eigenvalue floor',xlabel='Reported variance multiplier',
                ylabel='Smallest corrected eigenvalue (dex$^2$)')
    axes[0].legend(frameon=False,fontsize=11)
    axes[1].set(title='B  |  Sensitivity of the 95% variance summary',xlabel='Reported variance multiplier',
                ylabel='Regularized $d_{0.95}$',yticks=np.arange(1,10))
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(ROOT/'figures'/'apogee_noise_sensitivity.png',dpi=240)
    plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--components',type=int,default=64)
    ap.add_argument('--max-iter',type=int,default=1200)
    ap.add_argument('--bootstrap',type=int,default=400)
    ap.add_argument('--replicates',type=int,default=20)
    ap.add_argument('--benchmark-n',type=int,default=6000)
    run(ap.parse_args())
