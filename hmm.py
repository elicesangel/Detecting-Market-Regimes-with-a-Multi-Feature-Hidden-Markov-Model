"""HMM v7 — Multi-Feature + PCA + BIC + 5 Diagnostic Plots"""

import warnings; warnings.filterwarnings('ignore')
import os, json, numpy as np, pandas as pd, yfinance as yf
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM

OUT = 'C:/Users/UsuarioHP/Desktop/hmm_output'
TICKERS = ['SPY','IWM','HYG','LQD']
K_RANGE = range(3,9)
SEEDS = 5


def collect():
    print("STEP 1: DATA")
    etf = yf.download(TICKERS, start='2005-01-01', auto_adjust=False, progress=False)
    cl, hi, lo = etf['Close'], etf['High'], etf['Low']
    for c in [cl, hi, lo]:
        if isinstance(c.columns, pd.MultiIndex): c.columns = c.columns.droplevel(0)
    for t in TICKERS: print(f"  {t}: {cl[t].dropna().index[0].date()} → {cl[t].dropna().index[-1].date()} ({len(cl[t].dropna())})")
    vix_r = yf.download('^VIX', start='2005-01-01', auto_adjust=True, progress=False)['Close']
    vix = vix_r.iloc[:,0] if isinstance(vix_r, pd.DataFrame) else vix_r; vix.name='VIX'
    v3m_r = yf.download('^VIX3M', start='2005-01-01', auto_adjust=True, progress=False)['Close']
    vix3m = v3m_r.iloc[:,0] if isinstance(v3m_r, pd.DataFrame) else v3m_r; vix3m.name='VIX3M'
    print(f"  VIX: {len(vix.dropna())} | VIX3M: {len(vix3m.dropna())}")
    df = pd.DataFrame(index=cl.index)
    df['SPY'],df['SPY_High'],df['SPY_Low'] = cl['SPY'],hi['SPY'],lo['SPY']
    df['IWM'],df['HYG'],df['LQD'] = cl['IWM'],cl['HYG'],cl['LQD']
    df['VIX'],df['VIX3M'] = vix, vix3m
    df = df.ffill(limit=3).dropna()
    print(f"  Combined: {len(df)} rows")
    return df


def features(df):
    print("STEP 2: FEATURES")
    f = pd.DataFrame(index=df.index)
    sr, ir = df['SPY'].pct_change(), df['IWM'].pct_change()
    rv = lambda s,w: s.rolling(w).std()*np.sqrt(252)
    f['SPY_Above50D']=(df['SPY']>df['SPY'].rolling(50).mean()).astype(int)
    f['SPY_Above200D']=(df['SPY']>df['SPY'].rolling(200).mean()).astype(int)
    f['SPY_Dist_MA200']=df['SPY']/df['SPY'].rolling(200).mean()-1
    f['Credit_Spread']=np.log(df['HYG'])-np.log(df['LQD'])
    for t,r,n in [('SPY',sr,'SPY'),('IWM',ir,'IWM')]:
        f[f'{n}_Daily_Ret']=r; f[f'{n}_21D_Ret']=df[t].pct_change(21)
        f[f'{n}_63D_Ret']=df[t].pct_change(63); f[f'{n}_126D_Ret']=df[t].pct_change(126)
    f['SPY_vs_IWM_21D']=f['SPY_21D_Ret']-f['IWM_21D_Ret']
    f['SPY_vs_IWM_63D']=f['SPY_63D_Ret']-f['IWM_63D_Ret']
    f['SPY_21D_RVol']=rv(sr,21); f['SPY_63D_RVol']=rv(sr,63)
    f['IWM_21D_RVol']=rv(ir,21); f['IWM_63D_RVol']=rv(ir,63)
    f['SPY_Range']=(df['SPY_High']-df['SPY_Low'])/df['SPY']
    f['SPY_VoV_20']=f['SPY_21D_RVol'].rolling(20).std()
    f['VIX_Level']=df['VIX']; f['VIX_1D_Chg']=df['VIX'].diff(1); f['VIX_5D_Chg']=df['VIX'].diff(5)
    f['VIX_to_RVol']=df['VIX']/(f['SPY_21D_RVol']*100); f['VIX3M_VIX']=df['VIX3M']/df['VIX']
    f['SPY_126D_DD']=df['SPY']/df['SPY'].rolling(126,min_periods=1).max()-1
    f['IWM_126D_DD']=df['IWM']/df['IWM'].rolling(126,min_periods=1).max()-1
    f['SPY_IWM_63D_Corr']=sr.rolling(63).corr(ir)
    f['IWM_Beta_SPY']=sr.rolling(63).cov(ir)/sr.rolling(63).var()
    f = f.dropna()
    print(f"  {f.shape[1]} features × {f.shape[0]} obs")
    return f


def do_pca(feat, target=0.95):
    print("STEP 3: PCA")
    X = feat.replace([np.inf,-np.inf],np.nan).dropna()
    idx = X.index; cols = X.columns
    sc = StandardScaler(); Xs = sc.fit_transform(X.values)
    pca = PCA(); Xp = pca.fit_transform(Xs)
    n = np.argmax(np.cumsum(pca.explained_variance_ratio_)>=target)+1
    print(f"  {feat.shape[1]} → {n} PCs ({np.cumsum(pca.explained_variance_ratio_)[n-1]:.1%})")
    return Xp[:,:n], idx, Xs, pca, n, cols


def fit_hmm(X):
    print("STEP 4: HMM")
    T,d = X.shape
    def bic(m,X):
        K=m.n_components; logL=m.score(X); p=(K-1)+K*(K-1)+K*2*d
        return -2*logL+p*np.log(T), logL
    res=[]; best={'bic':np.inf,'m':None,'K':None,'logL':None}
    for K in K_RANGE:
        bk={'bic':np.inf,'m':None,'logL':None}
        for s in range(SEEDS):
            try:
                m=GaussianHMM(n_components=K,covariance_type='diag',n_iter=1000,random_state=42+s,verbose=False)
                m.fit(X); b,l=bic(m,X)
                if b<bk['bic']: bk={'bic':b,'m':m,'logL':l}
            except: pass
        res.append((K,bk['bic'],bk['logL']))
        if bk['bic']<best['bic']: best={'bic':bk['bic'],'m':bk['m'],'K':K,'logL':bk['logL']}
        print(f"  K={K}: BIC={bk['bic']:>12,.0f}")
    print(f"  Selected K={best['K']}")
    return best['m'], best['m'].predict(X), res


def diagnose(df, feat, states, idx, model):
    print("STEP 5: DIAGNOSTICS")
    K=model.n_components; ret=df['SPY'].pct_change().reindex(idx).values
    rv=feat['SPY_21D_RVol'].reindex(idx).values; dd=feat['SPY_126D_DD'].reindex(idx).values
    rows=[]
    for s in range(K):
        m=(states==s); r=ret[m]
        rows.append({'state':s,'n_days':int(m.sum()),'pct':f"{m.sum()/len(states)*100:.1f}%",
            'ann_ret':np.nanmean(r)*252*100,'ann_vol':np.nanmean(rv[m])*100,
            'avg_dd':np.nanmean(dd[m])*100,
            'sharpe':(np.nanmean(r)*252)/(np.nanstd(r)*np.sqrt(252)) if np.nanstd(r)>0 else 0})
    diag=pd.DataFrame(rows).sort_values('ann_ret',ascending=False).reset_index(drop=True)
    for _,r in diag.iterrows():
        print(f"  S{r['state']}: {r['n_days']}d ret={r['ann_ret']:.1f}% vol={r['ann_vol']:.1f}% sharpe={r['sharpe']:.2f}")
    return diag


def filtered(model, X):
    ls=np.log(np.maximum(model.startprob_,1e-300))
    lt=np.log(np.maximum(model.transmat_,1e-300))
    lb=model._compute_log_likelihood(X); T,K=lb.shape
    la=np.empty((T,K)); la[0]=ls+lb[0]; la[0]-=np.logaddexp.reduce(la[0])
    for t in range(1,T):
        tr=la[t-1][:,None]+lt; la[t]=np.logaddexp.reduce(tr,axis=0)+lb[t]
        la[t]-=np.logaddexp.reduce(la[t])
    p=np.exp(la[-1]); return p/p.sum()


# ═══════════════════════════════════════════════
# 5 DIAGNOSTIC PLOTS
# ═══════════════════════════════════════════════
def plot_correlation(Xs, cols, save_dir):
    print("  Plot 1/5: Correlation matrix")
    corr = np.corrcoef(Xs.T)
    fig, ax = plt.subplots(figsize=(16,14))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7); ax.set_yticklabels(cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v=corr[i,j]
            if abs(v)>0.5: ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=5.5,color='white' if abs(v)>0.7 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.75)
    mask=np.triu(np.ones_like(corr,dtype=bool),k=1); high=np.sum(np.abs(corr[mask])>0.7)
    ax.set_title(f'Feature Correlation Matrix — {high} pairs with |r| > 0.7', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir,'plot1_correlation.png'),dpi=150,bbox_inches='tight'); plt.close()


def plot_scree(pca, npc, save_dir):
    print("  Plot 2/5: PCA scree")
    ve = pca.explained_variance_ratio_; cv = np.cumsum(ve)
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
    a1.bar(range(1,len(ve)+1),ve*100,color='#1e40af',alpha=.7,edgecolor='white')
    a1.set_xlabel('Component'); a1.set_ylabel('Variance (%)'); a1.set_title('Individual')
    a2.plot(range(1,len(cv)+1),cv*100,'o-',color='#9f1239',lw=2,ms=5)
    a2.axhline(95,color='gray',ls='--',alpha=.5); a2.axvline(npc,color='#0d9488',ls='--',alpha=.5)
    a2.fill_between(range(1,npc+1),0,cv[:npc]*100,alpha=.12,color='#0d9488')
    a2.text(npc+.5,50,f'{npc} PCs = {cv[npc-1]:.1%}',fontsize=11,color='#0d9488',fontweight='bold')
    a2.set_xlabel('Components'); a2.set_ylabel('Cumulative (%)'); a2.set_title('Cumulative — 95% threshold')
    a2.set_ylim(0,105)
    plt.suptitle('PCA Variance Explained',fontsize=14,fontweight='bold'); plt.tight_layout(rect=[0,0,1,.95])
    plt.savefig(os.path.join(save_dir,'plot2_pca_scree.png'),dpi=150,bbox_inches='tight'); plt.close()


def plot_bic(bic_res, save_dir):
    print("  Plot 3/5: BIC selection")
    ks=[r[0] for r in bic_res]; bics=[r[1] for r in bic_res]; lls=[r[2] for r in bic_res]
    best_k=ks[np.argmin(bics)]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
    a1.plot(ks,bics,'o-',color='#9f1239',lw=2,ms=8)
    a1.scatter([best_k],[min(bics)],color='#0d9488',s=150,zorder=5,edgecolor='white',lw=2)
    for k,b in zip(ks,bics): a1.annotate(f'{b/1000:.1f}k',(k,b),textcoords="offset points",xytext=(0,12),ha='center',fontsize=9)
    a1.set_xlabel('K'); a1.set_ylabel('BIC'); a1.set_title('BIC (lower = better)')
    a2.plot(ks,lls,'o-',color='#1e40af',lw=2,ms=8)
    a2.set_xlabel('K'); a2.set_ylabel('Log-L'); a2.set_title('Log-Likelihood')
    plt.suptitle('Model Selection',fontsize=14,fontweight='bold'); plt.tight_layout(rect=[0,0,1,.95])
    plt.savefig(os.path.join(save_dir,'plot3_bic.png'),dpi=150,bbox_inches='tight'); plt.close()


def plot_diagnostics(diag, model, save_dir):
    print("  Plot 4/5: State diagnostics")
    order=diag['state'].tolist(); K=len(order)
    cm=plt.cm.RdYlGn(np.linspace(1,0,K)); sc={s:cm[i] for i,s in enumerate(order)}
    fig,axes=plt.subplots(2,2,figsize=(16,10))
    for ax,(col,title) in zip(axes.flat,[('ann_ret','Ann Return (%)'),('ann_vol','Ann Vol (%)'),('sharpe','Sharpe'),('avg_dd','Avg Drawdown (%)')]):
        d=diag.sort_values('state'); bars=ax.bar([f'S{s}' for s in d['state']],d[col],color=[sc[s] for s in d['state']],edgecolor='white',lw=.5)
        ax.set_title(title); ax.axhline(0,color='black',lw=.5); ax.grid(True,axis='y',alpha=.3)
        for b,v in zip(bars,d[col]): ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{v:.1f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=8)
    plt.suptitle(f'State Diagnostics (K={K})',fontsize=14,fontweight='bold'); plt.tight_layout(rect=[0,0,1,.95])
    plt.savefig(os.path.join(save_dir,'plot4_diagnostics.png'),dpi=150,bbox_inches='tight'); plt.close()


def plot_regimes_colors(df, states, idx, diag, model, save_dir):
    print("  Plot 5/6: Regime timeline — colours only")
    K=model.n_components; dates=idx; order=diag['state'].tolist()
    cm=plt.cm.RdYlGn(np.linspace(1,0,K)); sc={s:cm[i] for i,s in enumerate(order)}
    ss=pd.Series(states,index=dates); spy=df['SPY'].reindex(dates)
    fig,ax=plt.subplots(figsize=(20,6))
    ax.plot(dates,spy,color='#2c3e50',lw=.8)
    for s in range(K):
        ar=diag.loc[diag['state']==s,'ann_ret'].values[0]
        ax.fill_between(dates,spy.min()*.93,spy.max()*1.07,where=(ss==s),alpha=.15,color=sc[s],label=f"S{s} ({ar:.0f}%/yr)")
    ax.set_title(f'SPY — HMM v7 ({K} Regimes, BIC-selected) — Market Regime Changes',fontsize=14,fontweight='bold')
    ax.set_ylabel('Price ($)'); ax.legend(loc='upper left',fontsize=8,ncol=min(K,4)); ax.set_xlim(dates[0],dates[-1])
    plt.tight_layout(); plt.savefig(os.path.join(save_dir,'plot5_regimes_colors.png'),dpi=150,bbox_inches='tight'); plt.close()


def plot_regimes_combined(df, feat, states, idx, diag, model, save_dir):
    print("  Plot 6/6: Regime timeline — combined")
    K=model.n_components; dates=idx; order=diag['state'].tolist()
    cm=plt.cm.RdYlGn(np.linspace(1,0,K)); sc={s:cm[i] for i,s in enumerate(order)}
    ss=pd.Series(states,index=dates); spy=df['SPY'].reindex(dates)
    fig,ax=plt.subplots(3,1,figsize=(20,14),gridspec_kw={'height_ratios':[3,1.2,1.2]})
    ax[0].plot(dates,spy,color='#2c3e50',lw=.8)
    for s in range(K):
        ar=diag.loc[diag['state']==s,'ann_ret'].values[0]
        ax[0].fill_between(dates,spy.min()*.93,spy.max()*1.07,where=(ss==s),alpha=.15,color=sc[s],label=f"S{s} ({ar:.0f}%/yr)")
    ax[0].set_title(f'SPY — HMM v7 ({K} Regimes, BIC-selected)',fontsize=14,fontweight='bold')
    ax[0].set_ylabel('Price ($)'); ax[0].legend(loc='upper left',fontsize=8,ncol=min(K,4)); ax[0].set_xlim(dates[0],dates[-1])
    ax[1].plot(dates,feat['SPY_21D_RVol'].reindex(dates)*100,color='#7c3aed',lw=.7)
    ax[1].axhline(20,color='gray',ls='--',alpha=.3); ax[1].set_ylabel('21d RVol (%)'); ax[1].set_xlim(dates[0],dates[-1])
    dd=feat['SPY_126D_DD'].reindex(dates)*100
    ax[2].fill_between(dates,dd,0,alpha=.3,color='red'); ax[2].plot(dates,dd,color='#c0392b',lw=.6)
    ax[2].set_ylabel('126d DD (%)'); ax[2].set_xlim(dates[0],dates[-1])
    plt.tight_layout(); plt.savefig(os.path.join(save_dir,'plot6_regimes_combined.png'),dpi=150,bbox_inches='tight'); plt.close()


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def run():
    os.makedirs(OUT, exist_ok=True)
    print("="*55)
    print("  HMM v7 — Multi-Feature + PCA + BIC")
    print("="*55)
    df = collect()
    feat = features(df)
    X, idx, Xs, pca, npc, cols = do_pca(feat)
    model, states, bic_res = fit_hmm(X)
    diag = diagnose(df, feat, states, idx, model)
    print("STEP 6: CURRENT REGIME")
    fp = filtered(model, X); cs = int(np.argmax(fp))
    print(f"  {idx[-1].date()}: State {cs} (p={fp[cs]:.3f})")
    for s in np.argsort(-fp): print(f"    S{s}: {fp[s]:.4f}")
    print("STEP 7: PLOTS")
    plot_correlation(Xs, cols, OUT)
    plot_scree(pca, npc, OUT)
    plot_bic(bic_res, OUT)
    plot_diagnostics(diag, model, OUT)
    plot_regimes_colors(df, states, idx, diag, model, OUT)
    plot_regimes_combined(df, feat, states, idx, diag, model, OUT)
    summary = {'version':'v7','n_feat':feat.shape[1],'n_pca':npc,'K':int(model.n_components),
        'bic':[(k,float(b),float(l)) for k,b,l in bic_res],
        'diagnostics':diag.to_dict(orient='records'),
        'current_state':cs,'probs':{f"S{i}":float(p) for i,p in enumerate(fp)}}
    with open(os.path.join(OUT,'hmm_v7_summary.json'),'w') as f: json.dump(summary,f,indent=2,default=str)
    pd.DataFrame({'date':idx,'state':states}).to_csv(os.path.join(OUT,'regimes_v7.csv'),index=False)
    print("DONE")
    return model, feat, states, idx, diag

if __name__=='__main__':
    run()
