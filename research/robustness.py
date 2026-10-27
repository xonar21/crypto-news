"""
Проверка устойчивости трёх кандидатов из фазы 2 по годам, без 2022 года и при жёстких издержках.

  K1. CPI: пробой пред-релизного диапазона, вход t+5 по импульсу, выход t+60.
  K2. CPI: «стрэддл» отложенными стоп-ордерами (буфер 0.15%), стоп на противоположной границе, выход t+60.
  K3. FOMC: продолжение движения первого часа, вход t+60, выход t+1440.

Запуск: python3 research/robustness.py --btc DIR --eth DIR --sol DIR --events events.json --out research/results
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy import stats

from event_study import load_events
from phase2 import Px, grouped_events, features

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return (stats.ttest_1samp(x, 0).statistic if len(x) >= 3 else np.nan), len(x)


def k1(F):
    """Возвращает Series доходностей (gross) по событиям CPI."""
    G = F[F.is_cpi]
    d = np.sign(G.r5)
    brk = ((d > 0) & (G.p5 > G.pre_hi)) | ((d < 0) & (G.p5 < G.pre_lo))
    gross = d * ((1 + G.r60) / (1 + G.r5) - 1)
    return pd.Series(gross[brk].values, index=G.t[brk])


def k2(px: Px, ge, buf=0.0015, kx=60, slip=0.0):
    """Стрэддл: доходности по событиям CPI. slip — доп. проскальзывание на входе (цена исполнения хуже уровня)."""
    res = {}
    for _, e in ge[ge.is_cpi].iterrows():
        t0 = e["t"]
        hi, lo = px.rng(t0, -30, 0)
        if np.isnan(hi):
            continue
        up, dn = hi * (1 + buf), lo * (1 - buf)
        i, j = px.idx(t0), px.idx(t0 + pd.Timedelta(minutes=kx))
        if j <= i or j >= len(px.o):
            continue
        trig = None
        for b in range(i, j):
            if px.h[b] >= up and px.l[b] <= dn:
                trig = None; break
            if px.h[b] >= up:
                trig = (+1, b, up * (1 + slip)); break
            if px.l[b] <= dn:
                trig = (-1, b, dn * (1 - slip)); break
        if trig is None:
            continue
        dirn, b, entry = trig
        stop = dn if dirn > 0 else up
        r = np.nan
        for c in range(b + 1, j):
            if dirn > 0 and px.l[c] <= stop:
                r = stop / entry - 1; break
            if dirn < 0 and px.h[c] >= stop:
                r = entry / stop - 1; break
        if np.isnan(r):
            r = (px.o[j] / entry - 1) * dirn
        res[t0] = r
    return pd.Series(res)


def k3(F):
    G = F[F.is_fomc]
    d = np.sign(G.r60)
    gross = d * ((1 + G.r1440) / (1 + G.r60) - 1)
    return pd.Series(gross.values, index=G.t)


def by_year(P, name, series_by_sym, cost):
    P(f"### {name}\n")
    P("| Год | " + " | ".join(f"{s[:3]} N / средн. % / hit" for s in SYMS) + " | портфель 1/3 каждого: средн. % / t / hit |")
    P("|---|" + "---|" * (len(SYMS) + 1))
    all_years = sorted(set().union(*[set(s.index.year) for s in series_by_sym.values()]))
    for yr in all_years + ["всего", "без 2022"]:
        cells = []
        port = []
        for sym in SYMS:
            s = series_by_sym[sym]
            if yr == "всего":
                x = s
            elif yr == "без 2022":
                x = s[s.index.year != 2022]
            else:
                x = s[s.index.year == yr]
            if len(x) == 0:
                cells.append("—"); continue
            cells.append(f"{len(x)} / {x.mean()*100:+.2f} / {(x>0).mean()*100:.0f}%")
            port.append(x)
        if port:
            # портфель: средняя доходность по инструментам на каждое событие
            pf = pd.concat(port, axis=1, sort=True).mean(axis=1)
            t, n = tstat(pf)
            cells.append(f"{pf.mean()*100:+.2f} / {t:+.2f} / {(pf>0).mean()*100:.0f}% (N={n})")
        P(f"| {yr} | " + " | ".join(cells) + " |")
    P("")
    # экономика: годовой результат портфеля при 1x капитале, net
    pf = pd.concat([series_by_sym[s] for s in SYMS], axis=1, sort=True).mean(axis=1)
    net = pf - cost
    P(f"Портфель, издержки {cost*100:.2f}%: сделок {len(net)}, средн. net {net.mean()*100:+.3f}%, сумма net за период {net.sum()*100:+.1f}%, "
      f"худшая сделка {net.min()*100:.2f}%, стд {net.std()*100:.2f}%, t(net)={tstat(net)[0]:+.2f}")
    eq = (1 + net).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    P(f"Макс. просадка кривой капитала (без плеча): {dd*100:.2f}%; доля прибыльных лет: "
      f"{(net.groupby(net.index.year).sum() > 0).mean()*100:.0f}%\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc", required=True); ap.add_argument("--eth", required=True); ap.add_argument("--sol", required=True)
    ap.add_argument("--events", required=True); ap.add_argument("--out", default="research/results")
    a = ap.parse_args()
    lines = []; P = lines.append
    px = {"BTCUSDT": Px(a.btc, "BTCUSDT"), "ETHUSDT": Px(a.eth, "ETHUSDT"), "SOLUSDT": Px(a.sol, "SOLUSDT")}
    ge = grouped_events(load_events(a.events))
    F = {s: features(px[s], ge) for s in SYMS}

    P("# Устойчивость кандидатов по годам\n")
    by_year(P, "K1. CPI: пробой диапазона, вход t+5, выход t+60 (издержки 0.10%)", {s: k1(F[s]) for s in SYMS}, 0.0010)
    by_year(P, "K2. CPI: стрэддл стоп-ордерами, буфер 0.15%, выход t+60 (издержки 0.10%)", {s: k2(px[s], ge) for s in SYMS}, 0.0010)
    P("Тот же стрэддл с проскальзыванием исполнения стоп-ордера 0.20% (цена входа хуже уровня) и издержками 0.10%:\n")
    by_year(P, "K2b. CPI: стрэддл, буфер 0.15%, slip 0.20%, выход t+60", {s: k2(px[s], ge, slip=0.002) for s in SYMS}, 0.0010)
    by_year(P, "K2c. CPI: стрэддл, буфер 0.15%, выход t+240 (издержки 0.10%)", {s: k2(px[s], ge, kx=240) for s in SYMS}, 0.0010)
    by_year(P, "K3. FOMC: продолжение первого часа, вход t+60, выход t+1440 (издержки 0.10%)", {s: k3(F[s]) for s in SYMS}, 0.0010)

    # K3: детали — зависит ли от величины первого часа и от направления
    P("### K3 детали (SOL): по направлению первого часа\n")
    G = F["SOLUSDT"][F["SOLUSDT"].is_fomc]
    for lab, m in [("первый час вверх", G.r60 > 0), ("первый час вниз", G.r60 < 0)]:
        g = np.sign(G.r60[m]) * ((1 + G.r1440[m]) / (1 + G.r60[m]) - 1)
        t, n = tstat(g)
        P(f"- {lab}: N={n}, средн. {g.mean()*100:+.2f}%, t={t:+.2f}, hit {(g>0).mean()*100:.0f}%")
    P("")

    txt = "\n".join(lines)
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "results_robustness.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
