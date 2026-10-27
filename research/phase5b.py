import argparse, numpy as np, pandas as pd
from scipy import stats
from phase5 import PxF
from phase4 import fomc_times, cell
ap=argparse.ArgumentParser(); ap.add_argument("--btc"); ap.add_argument("--eth"); ap.add_argument("--sol"); ap.add_argument("--out", default="results"); a=ap.parse_args()
px={"BTCUSDT":PxF(a.btc,"BTCUSDT"),"ETHUSDT":PxF(a.eth,"ETHUSDT"),"SOLUSDT":PxF(a.sol,"SOLUSDT")}
L=[];P=L.append
P("# Уточнение фазы 5: «тихая» реакция на заявление FOMC\n")
P("volx = объём 14:00-14:30 / средний получасовой объём за предыдущие сутки. Вход 14:30 по знаку реакции, выход +24ч.\n")
F=fomc_times()
for s,p in px.items():
    R=F.copy(); R["r"]=[p.ret(t,0,30) for t in R.t]; R["x"]=[p.ret(t,30,1440) for t in R.t]; R["x48"]=[p.ret(t,30,2880) for t in R.t]
    R["volx"]=[p.vol(t,0,30)/(p.vol(t,-1440,0)/48) for t in R.t]; d=np.sign(R.r); R["g"]=d*R.x; R["y"]=R.t.dt.year
    P(f"## {s}\n"); P(f"- Spearman(volx, |r_stmt|) = {stats.spearmanr(R.volx, R.r.abs()).statistic:+.2f}; Spearman(volx, продолжение) = {stats.spearmanr(R.volx, R.g).statistic:+.2f} (p={stats.spearmanr(R.volx, R.g).pvalue:.2f})")
    P("| Разрез | N | +24ч | +48ч |"); P("|---|---|---|---|")
    for lab,m in [("volx < 3",R.volx<3),("volx 3-5",(R.volx>=3)&(R.volx<5)),("volx >= 5",R.volx>=5),
                  ("volx<3 и |r|>0.3%",(R.volx<3)&(R.r.abs()>0.003)),("volx<3 и |r|<=0.3%",(R.volx<3)&(R.r.abs()<=0.003)),
                  ("volx>=5 и |r|>0.3%",(R.volx>=5)&(R.r.abs()>0.003))]:
        g=R[m]; P(f"| {lab} | {len(g)} | {cell(g.g)} | {cell(np.sign(g.r)*g.x48)} |")
    P("\nvolx<3 по годам: "+"; ".join(f"{y}: {cell(g.g)}" for y,g in R[R.volx<3].groupby('y')))
    P("volx>=5 по годам: "+"; ".join(f"{y}: {cell(g.g)}" for y,g in R[R.volx>=5].groupby('y'))+"\n")
    P(f"Медиана volx по годам: {R.groupby('y').volx.median().round(1).to_dict()}\n")
txt="\n".join(L); open(f"{a.out}/results_phase5b.md","w").write(txt); print(txt)
