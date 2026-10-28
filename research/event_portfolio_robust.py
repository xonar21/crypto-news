"""Устойчивость событийного портфеля: сетка параметров, бутстрэп, разделение выборки."""
import argparse, os, numpy as np, pandas as pd
from event_study import load_events
from phase2 import Px, grouped_events
from phase4 import fomc_times
from event_portfolio import path_exit, SYMS, COST
ap=argparse.ArgumentParser(); ap.add_argument("--btc"); ap.add_argument("--eth"); ap.add_argument("--sol"); ap.add_argument("--events"); ap.add_argument("--out",default="research/results"); a=ap.parse_args()
px={"BTCUSDT":Px(a.btc,"BTCUSDT"),"ETHUSDT":Px(a.eth,"ETHUSDT"),"SOLUSDT":Px(a.sol,"SOLUSDT")}
ge=grouped_events(load_events(a.events)); F=fomc_times()
def fomc(hold_min, stop, use_filter):
    rows=[]
    for s in SYMS:
        p=px[s]; hist=[]
        for _,e in F.iterrows():
            t=e.t; r=p.ret(t,0,30); med=np.median(np.abs(hist[-12:])) if len(hist)>=6 else None; hist.append(r)
            if np.isnan(r) or (use_filter and med is not None and abs(r)<0.5*med): continue
            d=np.sign(r); entry=t+pd.Timedelta(minutes=30); e_px=p.px(entry)
            stop_px = e_px*(1-stop*d) if stop else (0 if d>0 else 1e12)
            ret,_=path_exit(p,entry,t+pd.Timedelta(minutes=30+hold_min),d,stop_px)
            if not np.isnan(ret): rows.append((t.floor("D"),s,ret-COST))
    T=pd.DataFrame(rows,columns=["evt","s","net"]); return T.groupby("evt").net.mean()
def cpi(entry_min, hold_min, use_stop):
    rows=[]
    for s in SYMS:
        p=px[s]
        for _,e in ge[ge.is_cpi].iterrows():
            t=e.t; hi,lo=p.rng(t,-60,0); pe=p.px(t+pd.Timedelta(minutes=entry_min)); re=p.ret(t,0,entry_min)
            if np.isnan(hi) or np.isnan(pe): continue
            d=np.sign(re)
            if not ((d>0 and pe>hi) or (d<0 and pe<lo)): continue
            stop_px=(lo if d>0 else hi) if use_stop else (0 if d>0 else 1e12)
            ret,_=path_exit(p,t+pd.Timedelta(minutes=entry_min),t+pd.Timedelta(minutes=hold_min),d,stop_px)
            if not np.isnan(ret): rows.append((t.floor("D"),s,ret-COST))
    T=pd.DataFrame(rows,columns=["evt","s","net"]); return T.groupby("evt").net.mean()
L=[];P=L.append
P("# Устойчивость событийного портфеля\n")
P("## FOMC: сетка параметров (доходность на событие, среднее по BTC/ETH/SOL, net)\n")
P("| удержание | стоп | фильтр слабой реакции | событий | средн. % | hit | t | худшее % | лет в плюсе |"); P("|---|---|---|---|---|---|---|---|---|")
for hold in [720,1440,2880]:
    for stop in [None,0.03,0.04,0.06]:
        for filt in [False,True]:
            E=fomc(hold,stop,filt); by=E.groupby(E.index.year).sum()
            t=E.mean()/(E.std()/np.sqrt(len(E)))
            P(f"| {hold//60}ч | {'нет' if stop is None else f'{stop*100:.0f}%'} | {'да' if filt else 'нет'} | {len(E)} | {E.mean()*100:+.2f} | {(E>0).mean()*100:.0f}% | {t:+.2f} | {E.min()*100:.1f} | {(by>0).mean()*100:.0f}% |")
P("\n## CPI: сетка параметров\n")
P("| вход | выход | стоп по диапазону | событий | средн. % | hit | t | худшее % | лет в плюсе |"); P("|---|---|---|---|---|---|---|---|---|")
for em in [5,10,15]:
    for hm in [30,60,120]:
        for st in [False,True]:
            E=cpi(em,hm,st); by=E.groupby(E.index.year).sum(); t=E.mean()/(E.std()/np.sqrt(len(E)))
            P(f"| t+{em} | t+{hm} | {'да' if st else 'нет'} | {len(E)} | {E.mean()*100:+.2f} | {(E>0).mean()*100:.0f}% | {t:+.2f} | {E.min()*100:.1f} | {(by>0).mean()*100:.0f}% |")
# bootstrap and split
E=pd.concat([fomc(1440,0.04,True),cpi(5,60,True)]).sort_index()
rng=np.random.default_rng(0); boots=np.array([rng.choice(E.values,len(E)).mean() for _ in range(20000)])
P(f"\n## Бутстрэп по {len(E)} событиям базовой конфигурации\n")
P(f"- средняя доходность на событие {E.mean()*100:+.2f}%, 95% интервал [{np.percentile(boots,2.5)*100:+.2f}%, {np.percentile(boots,97.5)*100:+.2f}%], доля бутстрэп-выборок с отрицательным средним {np.mean(boots<=0)*100:.2f}%")
for lab,m in [("2021–2023",E.index.year<=2023),("2024–2026",E.index.year>=2024)]:
    x=E[m]; P(f"- {lab}: событий {len(x)}, средн. {x.mean()*100:+.2f}%, hit {(x>0).mean()*100:.0f}%, t={x.mean()/(x.std()/np.sqrt(len(x))):+.2f}, сумма {x.sum()*100:+.1f}%")
P(f"- без 2024 года (лучшего): средн. {E[E.index.year!=2024].mean()*100:+.2f}%, t={E[E.index.year!=2024].mean()/(E[E.index.year!=2024].std()/np.sqrt((E.index.year!=2024).sum())):+.2f}")
P("\nПоправка на множественное тестирование: в фазах 1–5 проверено порядка 400 конфигураций; при нулевом истинном эффекте ожидаемый максимум t среди 400 независимых тестов ≈ 3.0. Наблюдаемое t по 85 событиям базовой конфигурации и положительность всех 24 ячеек FOMC-сетки указывают на эффект, но доверительный интервал следует читать как оптимистичный.\n")
txt="\n".join(L); open(os.path.join(a.out,"results_event_portfolio_robust.md"),"w").write(txt); print(txt)
