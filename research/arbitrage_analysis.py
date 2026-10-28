"""Разные коэффициенты у разных контор: вилки и value по честной цене Pinnacle. АПЛ 2002-2026."""
import argparse, os, numpy as np, pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument("--odds"); ap.add_argument("--out", default="research/results"); a=ap.parse_args()
od=pd.read_csv(a.odds, parse_dates=["date"]); res=pd.read_csv(os.path.join(os.path.dirname(a.odds),"results.csv"))[["match_id","season","ftr"]]
d=res.merge(od.drop(columns=["season"],errors="ignore"),on="match_id").sort_values("date").reset_index(drop=True)
d["y"]=d.ftr.map({"H":0,"D":1,"A":2})
books=sorted({c.split("_1x2_")[0] for c in d.columns if "_1x2_home" in c and not c.endswith("_close")} - {"market_max","market_avg","betbrain_max","betbrain_avg","betfair_ex"})
L=[];P=L.append
P("# Разные коэффициенты у разных контор: вилки и value по Pinnacle (АПЛ 2002–2026)\n")
P(f"Конторы в данных (открытие линий, один снимок на матч): {', '.join(books)}.\n")
# --- 1. вилки
P("## 1. Вилки (surebets): ставка на все три исхода в лучших конторах\n")
P("| Сезон | матчей | контор в среднем | доля матчей с вилкой | средняя прибыль вилки | макс. прибыль | доля с вилкой > 1% |")
P("|---|---|---|---|---|---|---|")
rows=[]
for s,g in d.groupby("season"):
    H=g[[f"{b}_1x2_home" for b in books]].to_numpy(float); D=g[[f"{b}_1x2_draw" for b in books]].to_numpy(float); A=g[[f"{b}_1x2_away" for b in books]].to_numpy(float)
    nb=(~np.isnan(H)).sum(1); ok=nb>=3
    if ok.sum()<50: continue
    bh,bd,ba=np.nanmax(H[ok],1),np.nanmax(D[ok],1),np.nanmax(A[ok],1)
    inv=1/bh+1/bd+1/ba; arb=inv<1; profit=np.where(arb,1/inv-1,0)
    rows.append((s,ok.sum(),nb[ok].mean(),arb.mean(),profit[arb].mean() if arb.any() else 0,profit.max(),(profit>0.01).mean()))
    P(f"| {s} | {ok.sum()} | {nb[ok].mean():.1f} | {arb.mean()*100:.1f}% | {profit[arb].mean()*100 if arb.any() else 0:.2f}% | {profit.max()*100:.2f}% | {(profit>0.01).mean()*100:.1f}% |")
P("")
# --- 2. Pinnacle-fair value
P("## 2. Value по честной цене: вероятность = де-вигованный Pinnacle (открытие), ставка там, где коэффициент другой конторы × p > порога\n")
m=d.pinnacle_1x2_home.notna()&d.pinnacle_1x2_draw.notna()&d.pinnacle_1x2_away.notna(); g=d[m].copy()
inv=1/g.pinnacle_1x2_home+1/g.pinnacle_1x2_draw+1/g.pinnacle_1x2_away
pf=np.column_stack([(1/g.pinnacle_1x2_home)/inv,(1/g.pinnacle_1x2_draw)/inv,(1/g.pinnacle_1x2_away)/inv]); y=g.y.to_numpy(); Y=np.eye(3)[y]
soft=[b for b in books if b!="pinnacle"]
P(f"Матчей с Pinnacle: {len(g)} (2012–2026). Каждая пара (контора, исход) — отдельная потенциальная ставка.\n")
P("| Порог EV | ставок | доля от возможных | выигрыш на ставку | t | доля сезонов в плюсе | ставок в год |")
P("|---|---|---|---|---|---|---|")
allrec=[]
for b in soft:
    O=g[[f"{b}_1x2_home",f"{b}_1x2_draw",f"{b}_1x2_away"]].to_numpy(float)
    ok=~np.isnan(O).any(1)
    if ok.sum()<100: continue
    ev=O*pf; 
    for i in np.where(ok)[0]:
        for k in range(3):
            allrec.append((g.season.iloc[i],b,k,ev[i,k],Y[i,k]*O[i,k]-1,O[i,k],pf[i,k]))
R=pd.DataFrame(allrec,columns=["season","book","k","ev","ret","odds","p"])
P(f"Всего пар (контора, исход): {len(R)}; средний EV по Pinnacle-честной цене {R.ev.mean():.4f} (т.е. средняя маржа мягких контор против Pinnacle {(1-R.ev.mean())*100:.1f}%).\n")
for thr in [1.00,1.02,1.05,1.10]:
    x=R[R.ev>thr]
    if len(x)<30: continue
    by=x.groupby("season").ret.sum(); t=x.ret.mean()/(x.ret.std()/np.sqrt(len(x)))
    P(f"| > {thr:.2f} | {len(x)} | {len(x)/len(R)*100:.1f}% | {x.ret.mean()*100:+.2f}% | {t:+.2f} | {(by>0).mean()*100:.0f}% | {len(x)/x.season.nunique():.0f} |")
P("")
P("Та же стратегия (EV > 1.02) по конторам:\n")
P("| Контора | ставок | выигрыш на ставку | t | средний коэф. |")
P("|---|---|---|---|---|")
for b,x in R[R.ev>1.02].groupby("book"):
    if len(x)<40: continue
    t=x.ret.mean()/(x.ret.std()/np.sqrt(len(x))); P(f"| {b} | {len(x)} | {x.ret.mean()*100:+.2f}% | {t:+.2f} | {x.odds.mean():.2f} |")
P("")
P("По исходам и по сезонам (EV > 1.02):\n")
for k,lab in enumerate(["хозяева","ничья","гости"]):
    x=R[(R.ev>1.02)&(R.k==k)]; t=x.ret.mean()/(x.ret.std()/np.sqrt(len(x))) if len(x)>1 else np.nan
    P(f"- {lab}: ставок {len(x)}, выигрыш на ставку {x.ret.mean()*100:+.2f}%, t={t:+.2f}")
x=R[R.ev>1.02]; by=x.groupby("season").agg(n=("ret","size"),ret=("ret","mean"))
P("- по сезонам: "+"; ".join(f"{s}: N={r.n}, {r.ret*100:+.1f}%" for s,r in by.iterrows()))
P("")
# --- 3. Соотношение с CLV: ставки с EV>1.02 — движется ли Pinnacle к закрытию в их сторону?
mc=g.pinnacle_1x2_home_close.notna()
if mc.sum()>500:
    gc=g[mc]; invc=1/gc.pinnacle_1x2_home_close+1/gc.pinnacle_1x2_draw_close+1/gc.pinnacle_1x2_away_close
    pc=np.column_stack([(1/gc.pinnacle_1x2_home_close)/invc,(1/gc.pinnacle_1x2_draw_close)/invc,(1/gc.pinnacle_1x2_away_close)/invc])
    pfo=pf[mc.to_numpy()]
    # для каждой мягкой конторы: ставки с EV>1.02 по открытию -> CLV относительно закрытия
    clvs=[]
    for b in soft:
        O=gc[[f"{b}_1x2_home",f"{b}_1x2_draw",f"{b}_1x2_away"]].to_numpy(float); ok=~np.isnan(O).any(1)
        ev=O*pfo; sel=(ev>1.02)&ok[:,None]
        ii,kk=np.where(sel); clvs.extend(list(pc[ii,kk]-pfo[ii,kk]))
    clvs=np.array(clvs)
    P(f"## 3. Закрывающая линия\n\nДля ставок с EV > 1.02 по открытию (2019–2026, N={len(clvs)}): сдвиг честной вероятности Pinnacle к закрытию в сторону ставки {clvs.mean()*100:+.2f} п.п. (t={clvs.mean()/(clvs.std()/np.sqrt(len(clvs))):+.1f}), доля с положительным CLV {(clvs>0).mean()*100:.0f}%.\n")
txt="\n".join(L); open(os.path.join(a.out,"results_arbitrage.md"),"w").write(txt); print(txt)
