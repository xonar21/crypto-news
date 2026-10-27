"""Уточнения фазы 4: вход по заявлению FOMC в 14:30 ET; выходные BTC -> понедельник S&P по годам."""
import argparse, os
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from scipy import stats
from phase2 import Px
from phase4 import fomc_times, cell, tstat
NY = ZoneInfo("America/New_York")
ap = argparse.ArgumentParser(); ap.add_argument("--btc"); ap.add_argument("--eth"); ap.add_argument("--sol"); ap.add_argument("--xasset"); ap.add_argument("--out", default="results")
a = ap.parse_args()
px = {"BTCUSDT": Px(a.btc, "BTCUSDT"), "ETHUSDT": Px(a.eth, "ETHUSDT"), "SOLUSDT": Px(a.sol, "SOLUSDT")}
L = []; P = L.append
P("# Уточнения фазы 4\n")
P("## FOMC: вход в 14:30 ET по направлению реакции на заявление (14:00-14:30) вместо ожидания часа\n")
F = fomc_times()
for s, p in px.items():
    R = F.copy()
    R["r_stmt"] = [p.ret(t, 0, 30) for t in R.t]; R["r_pre24"] = [p.ret(t, -1440, 0) for t in R.t]
    for k in [240, 1440, 2880]:
        R[f"x{k}"] = [p.ret(t, 30, k) for t in R.t]
    d = np.sign(R.r_stmt)
    P(f"### {s}\n")
    P("| Разрез | N | +4ч | +24ч | +48ч |")
    P("|---|---|---|---|---|")
    for lab, m in [("все", R.r_stmt.notna()), ("|r_stmt| > 0.5%", R.r_stmt.abs() > 0.005), ("|r_stmt| <= 0.5%", R.r_stmt.abs() <= 0.005),
                   ("заявление против суток до", np.sign(R.r_stmt) != np.sign(R.r_pre24)), ("заявление по суткам до", np.sign(R.r_stmt) == np.sign(R.r_pre24)),
                   ("пауза", R.chg == 0), ("изменение ставки", R.chg != 0)]:
        g = R[m]; dd = np.sign(g.r_stmt)
        P(f"| {lab} | {len(g)} | {cell(dd*g.x240)} | {cell(dd*g.x1440)} | {cell(dd*g.x2880)} |")
    P("")
    P("По годам (вход 14:30, выход +24ч): " + "; ".join(f"{y}: {cell(np.sign(g.r_stmt)*g.x1440)}" for y, g in R.groupby(R.t.dt.year)))
    P("")
# weekend -> Monday SPX
P("## Выходные крипты → понедельник S&P 500: по годам и с окном до открытия фьючерсов\n")
spx = pd.read_csv(os.path.join(a.xasset, "us_market.csv"), parse_dates=["Date"]).set_index("Date")["Close"].pct_change()
spx = spx[spx.index >= "2021-01-01"]
rows = []
for d in spx.index:
    if d.weekday() != 0: continue
    fri = d - pd.Timedelta(days=3)
    def at(day, hm): return px["BTCUSDT"].px(pd.Timestamp(f"{day.date()} {hm}", tz=NY).tz_convert("UTC"))
    p_fri, p_sun18, p_mon930 = at(fri, "16:00"), at(d - pd.Timedelta(days=1), "18:00"), at(d, "09:30")
    if np.isnan(p_fri) or np.isnan(p_mon930): continue
    rows.append({"d": d, "y": d.year, "wk_full": p_mon930 / p_fri - 1, "wk_pre_fut": p_sun18 / p_fri - 1, "spx": spx[d]})
W = pd.DataFrame(rows)
for lab, col in [("пт 16:00 → пн 09:30 ET", "wk_full"), ("пт 16:00 → вс 18:00 ET (до открытия фьючерсов)", "wk_pre_fut")]:
    sp = stats.spearmanr(W[col], W.spx)
    P(f"- {lab}: N={len(W)}, Spearman={sp.statistic:+.3f} (p={sp.pvalue:.3f}); S&P пн после BTC < −3%: {cell(W.spx[W[col] < -0.03])}; после > +3%: {cell(W.spx[W[col] > 0.03])}; "
      f"long/short S&P по знаку BTC-выходных при |BTC| > 2%: {cell(np.sign(W[col][W[col].abs() > 0.02]) * W.spx[W[col].abs() > 0.02])}")
P("")
P("| Год | N | Spearman (полные выходные) | S&P пн по знаку BTC-выходных при |BTC|>2%: средн. % / t / hit |")
P("|---|---|---|---|")
for y, g in W.groupby("y"):
    sp = stats.spearmanr(g.wk_full, g.spx); m = g.wk_full.abs() > 0.02
    P(f"| {y} | {len(g)} | {sp.statistic:+.2f} (p={sp.pvalue:.2f}) | {cell(np.sign(g.wk_full[m]) * g.spx[m])} |")
txt = "\n".join(L); open(os.path.join(a.out, "results_phase4b.md"), "w").write(txt); print(txt)
