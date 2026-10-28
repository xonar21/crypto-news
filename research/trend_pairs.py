"""
Систематические правила на корзине из 15 крупных монет (часовые бары 2021-2026), издержки учтены.

  A. Временной тренд (time-series momentum): лонг, если цена выше своей скользящей средней / доходность за lookback > 0,
     иначе вне рынка (long/flat) или шорт (long/short). Ребаланс раз в день. Равные веса по монетам.
  B. Кросс-секционный моментум: раз в неделю лонг топ-3 по доходности за 30 дней, шорт боттом-3.
  C. Пары: спред log(ETH/BTC) (и SOL/BTC) с z-score по 30 дням, вход при |z|>2, выход при z=0, стоп |z|>4.
  D. Волатильностный таргетинг как оболочка: размер позиции = целевая вол / реализованная вол.

Честность: параметры заданы заранее стандартными из литературы (Moskowitz-Ooi-Pedersen для TSMOM, Liu-Tsyvinski-Wu для крипты),
без перебора; отчёт по годам; издержки 0.10% на оборот.

Запуск: python3 research/trend_pairs.py --hourly DIR --out research/results
"""
import argparse, glob, os
import numpy as np, pandas as pd

COST = 0.001


def load(hdir):
    out = {}
    for f in sorted(glob.glob(os.path.join(hdir, "*_1h.csv"))):
        s = os.path.basename(f).split("_")[0]
        d = pd.read_csv(f, parse_dates=["t"]).drop_duplicates("t").set_index("t").sort_index()
        out[s] = d["close"]
    px = pd.DataFrame(out)
    return px


def perf(ret, label, P, periods_per_year=365):
    ret = ret.dropna()
    if len(ret) < 50:
        P(f"| {label} | — | — | — | — | — | — |"); return
    eq = (1 + ret).cumprod(); yrs = len(ret) / periods_per_year
    cagr = eq.iloc[-1] ** (1 / yrs) - 1; vol = ret.std() * np.sqrt(periods_per_year); dd = (eq / eq.cummax() - 1).min()
    by = ret.groupby(ret.index.year).apply(lambda x: (1 + x).prod() - 1)
    P(f"| {label} | {cagr*100:+.1f}% | {vol*100:.1f}% | {cagr/vol if vol>0 else np.nan:+.2f} | {dd*100:.1f}% | {(by>0).mean()*100:.0f}% | " + " / ".join(f"{y}:{v*100:+.0f}" for y, v in by.items()) + " |")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hourly", required=True); ap.add_argument("--out", default="research/results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    px = load(a.hourly)
    daily = px.resample("1D").last().dropna(how="all")
    rets = daily.pct_change()
    L = []; P = L.append
    P(f"# Трендовые и парные правила на {px.shape[1]} монетах, {daily.index.min().date()} — {daily.index.max().date()}\n")
    P(f"Монеты: {', '.join(px.columns)}. Издержки {COST*100:.2f}% на единицу оборота.\n")
    HDR = "| Правило | CAGR | вол | Sharpe | макс. просадка | доля лет в плюсе | по годам |\n|---|---|---|---|---|---|---|"

    # Benchmarks
    P("## Бенчмарки\n"); P(HDR)
    perf(rets["BTCUSDT"], "BTC buy-and-hold", P)
    perf(rets.mean(axis=1), "равновзвешенная корзина buy-and-hold (ежедневный ребаланс)", P)
    P("")

    # A. TSMOM
    P("## A. Временной тренд, равные веса по монетам, ребаланс ежедневно\n"); P(HDR)
    for lb in [20, 50, 100, 200]:
        sig = (daily > daily.rolling(lb).mean()).astype(float).shift(1)   # сигнал по вчерашнему закрытию
        pos = sig.div(sig.shape[1])
        gross = (pos * rets).sum(axis=1); turn = pos.diff().abs().sum(axis=1)
        perf(gross - COST * turn, f"long/flat, цена > SMA{lb}", P)
    for lb in [30, 90]:
        sig = np.sign(daily.pct_change(lb)).shift(1); pos = sig.div(sig.shape[1])
        gross = (pos * rets).sum(axis=1); turn = pos.diff().abs().sum(axis=1)
        perf(gross - COST * turn, f"long/short по знаку доходности за {lb} дней", P)
        sigf = (sig > 0).astype(float); posf = sigf.div(sigf.shape[1])
        perf((posf * rets).sum(axis=1) - COST * posf.diff().abs().sum(axis=1), f"long/flat по знаку доходности за {lb} дней", P)
    # BTC only
    sig = (daily["BTCUSDT"] > daily["BTCUSDT"].rolling(100).mean()).astype(float).shift(1)
    perf(sig * rets["BTCUSDT"] - COST * sig.diff().abs(), "BTC long/flat, цена > SMA100", P)
    P("")

    # D. vol targeting wrapper on the best-known: long/flat SMA100
    P("## D. То же с волатильностным таргетингом (цель 20% годовых, вол за 30 дней)\n"); P(HDR)
    sig = (daily > daily.rolling(100).mean()).astype(float).shift(1)
    rv = rets.rolling(30).std() * np.sqrt(365)
    w = (0.20 / rv).clip(upper=2.0) * sig; w = w.div(w.shape[1])
    gross = (w * rets).sum(axis=1); turn = w.diff().abs().sum(axis=1)
    perf(gross - COST * turn, "long/flat SMA100 + vol-target 20%", P)
    P(f"- средняя суммарная экспозиция: {w.sum(axis=1).mean():.2f} капитала; средний дневной оборот {turn.mean()*100:.1f}% капитала\n")

    # B. cross-sectional momentum
    P("## B. Кросс-секционный моментум (еженедельно: лонг топ-3, шорт боттом-3 по доходности за 30 дней)\n"); P(HDR)
    wk = daily.resample("W-MON").last(); wrets = wk.pct_change()
    for lb_w, lab in [(4, "4 недели"), (13, "13 недель")]:
        mom = wk.pct_change(lb_w).shift(1)
        rank = mom.rank(axis=1)
        n = rank.notna().sum(axis=1)
        long = (rank.ge(n.values[:, None] - 2)).astype(float); short = (rank.le(3)).astype(float)
        pos = (long / 3 - short / 3).where(n >= 8, 0)
        gross = (pos * wrets).sum(axis=1); turn = pos.diff().abs().sum(axis=1)
        perf(gross - COST * turn, f"L/S моментум {lab}", P, periods_per_year=52)
        posl = (long / 3).where(n >= 8, 0)
        perf((posl * wrets).sum(axis=1) - COST * posl.diff().abs().sum(axis=1), f"только лонг топ-3, {lab}", P, periods_per_year=52)
    P("")

    # C. pairs
    P("## C. Парный спред к BTC (часовые бары): z-score log-отношения по 30 дням (720 часов)\n")
    P("| Пара | сделок | средн. net на сделку | hit | сумма net | макс. убыток сделки | по годам (сумма) |\n|---|---|---|---|---|---|---|")
    for alt in ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT"]:
        if alt not in px: continue
        s = np.log(px[alt] / px["BTCUSDT"]).dropna()
        m = s.rolling(720).mean(); sd = s.rolling(720).std(); z = (s - m) / sd
        trades = []; pos = 0; entry = None
        zz = z.to_numpy(); ss = s.to_numpy(); idx = s.index
        for i in range(720, len(zz)):
            if pos == 0:
                if zz[i] > 2: pos = -1; entry = i
                elif zz[i] < -2: pos = 1; entry = i
            else:
                if (pos == 1 and zz[i] >= 0) or (pos == -1 and zz[i] <= 0) or abs(zz[i]) > 4 or i - entry > 24 * 30:
                    r = pos * (ss[i] - ss[entry]) - 2 * COST   # спред = лонг одной ноги, шорт другой: 2 ноги
                    trades.append((idx[entry], r)); pos = 0
        T = pd.DataFrame(trades, columns=["t", "r"])
        if len(T) < 5: continue
        by = T.groupby(T.t.dt.year).r.sum()
        P(f"| {alt}/BTC | {len(T)} | {T.r.mean()*100:+.2f}% | {(T.r>0).mean()*100:.0f}% | {T.r.sum()*100:+.1f}% | {T.r.min()*100:.1f}% | " + " / ".join(f"{y}:{v*100:+.0f}" for y, v in by.items()) + " |")
    P("")
    txt = "\n".join(L); open(os.path.join(a.out, "results_trend_pairs.md"), "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
