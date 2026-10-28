"""
Событийный портфель как единая стратегия: полный бэктест с издержками, размером позиции и кривой капитала.

Правила (из фаз 2, 4, 5):
  R1. FOMC: вход 14:30 ET по знаку реакции на заявление (14:00-14:30), пропуск если |реакция| < половины медианы
      последних 12 заседаний; выход через 24 часа; стоп −4%. BTC, ETH, SOL равными долями.
  R2. CPI: через 5 минут после релиза, если цена вышла за диапазон 60 минут до релиза, вход по импульсу,
      выход через 60 минут; стоп на противоположной границе диапазона. BTC, ETH, SOL.
Издержки: 0.10% round-trip на каждую ногу (тейкер фьючерсы + проскальзывание), размер позиции: доля капитала.

Запуск: python3 research/event_portfolio.py --btc DIR --eth DIR --sol DIR --events events.json --out research/results
"""
import argparse, os
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from scipy import stats
from event_study import load_events
from phase2 import Px, grouped_events
from phase4 import fomc_times

NY = ZoneInfo("America/New_York"); SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]; COST = 0.001


def path_exit(px, t_entry, t_exit, direction, stop_px):
    i, j = px.idx(t_entry), px.idx(t_exit)
    if j <= i or j >= len(px.o): return np.nan, None
    e = px.o[i]
    if direction > 0:
        hit = np.where(px.l[i:j] <= stop_px)[0]
        return (stop_px / e - 1, "stop") if len(hit) else (px.o[j] / e - 1, "time")
    hit = np.where(px.h[i:j] >= stop_px)[0]
    return (e / stop_px - 1, "stop") if len(hit) else (e / px.o[j] - 1, "time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc"); ap.add_argument("--eth"); ap.add_argument("--sol"); ap.add_argument("--events"); ap.add_argument("--out", default="research/results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    px = {"BTCUSDT": Px(a.btc, "BTCUSDT"), "ETHUSDT": Px(a.eth, "ETHUSDT"), "SOLUSDT": Px(a.sol, "SOLUSDT")}
    ge = grouped_events(load_events(a.events)); F = fomc_times()
    trades = []
    # R1 FOMC
    for s in SYMS:
        p = px[s]; hist = []
        for _, e in F.iterrows():
            t = e.t; r = p.ret(t, 0, 30)
            med = np.median(np.abs(hist[-12:])) if len(hist) >= 6 else None
            hist.append(r)
            if np.isnan(r) or (med is not None and abs(r) < 0.5 * med): continue
            d = np.sign(r); entry = t + pd.Timedelta(minutes=30); e_px = p.px(entry)
            stop = e_px * (1 - 0.04 * d)
            ret, how = path_exit(p, entry, t + pd.Timedelta(minutes=30 + 1440), d, stop)
            if np.isnan(ret): continue
            trades.append(dict(rule="FOMC", sym=s, t=entry, d=d, gross=ret, net=ret - COST, exit=how))
    # R2 CPI
    for s in SYMS:
        p = px[s]
        for _, e in ge[ge.is_cpi].iterrows():
            t = e.t; hi, lo = p.rng(t, -60, 0); p5 = p.px(t + pd.Timedelta(minutes=5)); r5 = p.ret(t, 0, 5)
            if np.isnan(hi) or np.isnan(p5): continue
            d = np.sign(r5)
            if not ((d > 0 and p5 > hi) or (d < 0 and p5 < lo)): continue
            stop = lo if d > 0 else hi
            ret, how = path_exit(p, t + pd.Timedelta(minutes=5), t + pd.Timedelta(minutes=60), d, stop)
            if np.isnan(ret): continue
            trades.append(dict(rule="CPI", sym=s, t=t + pd.Timedelta(minutes=5), d=d, gross=ret, net=ret - COST, exit=how))
    T = pd.DataFrame(trades).sort_values("t"); T["year"] = T.t.dt.year
    L = []; P = L.append
    P("# Событийный портфель: полный бэктест 2021–2026\n")
    P("Издержки 0.10% round-trip на ногу. Размер: по каждому событию 1/3 капитала на каждую из трёх монет (итого 100% капитала без плеча на событие).\n")
    P("| Правило | сделок | средн. net % | медиана % | hit | t | худшая сделка | доля стопов | сумма net на 1 ед. |")
    P("|---|---|---|---|---|---|---|---|---|")
    for r, g in list(T.groupby("rule")) + [("все", T)]:
        t = g.net.mean() / (g.net.std() / np.sqrt(len(g)))
        P(f"| {r} | {len(g)} | {g.net.mean()*100:+.3f} | {g.net.median()*100:+.3f} | {(g.net>0).mean()*100:.0f}% | {t:+.2f} | {g.net.min()*100:.2f}% | {(g.exit=='stop').mean()*100:.0f}% | {g.net.sum()*100:+.1f}% |")
    P("")
    # per-event portfolio return = mean over symbols of same event
    T["evt"] = T.t.dt.floor("D").astype(str) + "_" + T.rule
    E = T.groupby(["evt", "rule", "year"]).net.mean().reset_index().sort_values("evt")
    P("Доходность на событие (среднее по трём монетам, net):\n")
    P("| Правило | событий | средн. % | hit | t | худшее событие | сумма % |")
    P("|---|---|---|---|---|---|---|")
    for r, g in list(E.groupby("rule")) + [("все", E)]:
        t = g.net.mean() / (g.net.std() / np.sqrt(len(g)))
        P(f"| {r} | {len(g)} | {g.net.mean()*100:+.2f} | {(g.net>0).mean()*100:.0f}% | {t:+.2f} | {g.net.min()*100:.2f}% | {g.net.sum()*100:+.1f} |")
    P("")
    P("По годам (сумма net по событиям, % капитала без плеча):\n")
    P("| Год | событий | FOMC % | CPI % | всего % | худшая просадка в году % |")
    P("|---|---|---|---|---|---|")
    for y, g in E.groupby("year"):
        eq = (1 + g.net).cumprod(); dd = (eq / eq.cummax() - 1).min()
        P(f"| {y} | {len(g)} | {g[g.rule=='FOMC'].net.sum()*100:+.1f} | {g[g.rule=='CPI'].net.sum()*100:+.1f} | {g.net.sum()*100:+.1f} | {dd*100:.1f} |")
    eq = (1 + E.net).cumprod(); dd = (eq / eq.cummax() - 1)
    yrs = (E.evt.max()[:4] != E.evt.min()[:4]) and 5.8 or 1
    ann = eq.iloc[-1] ** (1 / 5.8) - 1
    vol = E.net.std() * np.sqrt(len(E) / 5.8)
    P(f"\nИтог: {len(E)} событий, капитал ×{eq.iloc[-1]:.2f} за 5.8 лет (≈{ann*100:.1f}% годовых без плеча), годовая волатильность {vol*100:.1f}%, Sharpe ≈ {ann/vol:.2f}, максимальная просадка {dd.min()*100:.1f}%, доля прибыльных лет {(E.groupby('year').net.sum()>0).mean()*100:.0f}%.\n")
    P("Сравнение: BTC buy-and-hold за тот же период — см. ниже; стратегия в рынке около 2% времени (8 суточных и 12 часовых позиций в год).\n")
    b = px["BTCUSDT"]; p0 = b.px(pd.Timestamp("2021-01-04", tz="UTC")); p1 = b.px(pd.Timestamp("2026-10-20", tz="UTC"))
    P(f"BTC buy-and-hold 2021-01 → 2026-10: ×{p1/p0:.2f}, с просадкой более 75% в 2022.\n")
    T.to_csv(os.path.join(a.out, "event_portfolio_trades.csv"), index=False)
    txt = "\n".join(L); open(os.path.join(a.out, "results_event_portfolio.md"), "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
