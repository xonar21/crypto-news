"""
Плацебо-тест: те же правила K1/K2/K3 на днях БЕЗ события, в то же время суток (тот же день недели).
Если эффект есть и на пустых днях, то это общий внутридневной моментум, а не влияние новости.
"""
import argparse, os
import numpy as np, pandas as pd
from scipy import stats
from event_study import load_events
from phase2 import Px, grouped_events

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return (stats.ttest_1samp(x, 0).statistic if len(x) >= 3 else np.nan), len(x)

def k3_like(px, t0):
    r60 = px.ret(t0, 0, 60); r1440 = px.ret(t0, 0, 1440)
    return np.sign(r60) * ((1 + r1440) / (1 + r60) - 1)

def k1_like(px, t0):
    hi, lo = px.rng(t0, -60, 0); p5 = px.px(t0 + pd.Timedelta(minutes=5)); r5 = px.ret(t0, 0, 5); r60 = px.ret(t0, 0, 60)
    d = np.sign(r5)
    if (d > 0 and p5 > hi) or (d < 0 and p5 < lo):
        return d * ((1 + r60) / (1 + r5) - 1)
    return np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc"); ap.add_argument("--eth"); ap.add_argument("--sol"); ap.add_argument("--events"); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    px = {"BTCUSDT": Px(a.btc, "BTCUSDT"), "ETHUSDT": Px(a.eth, "ETHUSDT"), "SOLUSDT": Px(a.sol, "SOLUSDT")}
    ge = grouped_events(load_events(a.events))
    ev_days = set(ge.t.dt.floor("D"))
    L = []; P = L.append
    P("# Плацебо: те же правила в те же часы на днях без макрособытий\n")
    for name, mask, fn in [("K3-подобное (продолжение первого часа на 24ч), FOMC vs среды без событий в 18:00/19:00 UTC", ge.is_fomc, k3_like),
                           ("K1-подобное (пробой 60м диапазона, вход t+5, выход t+60), CPI vs дни без событий в 12:30/13:30 UTC", ge.is_cpi, k1_like)]:
        P(f"## {name}\n")
        P("| Инструмент | событие: N / средн. % / t / hit | плацебо: N / средн. % / t / hit |")
        P("|---|---|---|")
        ev = ge[mask]
        for s in SYMS:
            real = np.array([fn(px[s], t) for t in ev.t])
            plc = []
            for t in ev.t:
                # те же часы в соседние недели (±1..±4 недели), исключая дни с любыми событиями
                for w in [-4, -3, -2, -1, 1, 2, 3, 4]:
                    tt = t + pd.Timedelta(weeks=w)
                    if tt.floor("D") in ev_days: continue
                    if tt < pd.Timestamp("2021-01-02", tz="UTC") or tt > pd.Timestamp("2026-10-20", tz="UTC"): continue
                    plc.append(fn(px[s], tt))
            plc = np.array(plc)
            tr, nr = tstat(real); tp, npl = tstat(plc)
            rr = real[~np.isnan(real)]; pp = plc[~np.isnan(plc)]
            P(f"| {s} | {nr} / {rr.mean()*100:+.2f} / {tr:+.2f} / {(rr>0).mean()*100:.0f}% | {npl} / {pp.mean()*100:+.2f} / {tp:+.2f} / {(pp>0).mean()*100:.0f}% |")
        P("")
    txt = "\n".join(L); open(os.path.join(a.out, "results_placebo.md"), "w").write(txt); print(txt)

if __name__ == "__main__":
    main()
