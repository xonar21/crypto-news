"""
Фаза 4: шире и глубже.

  A. Вселенная шоков без ручного отбора: все часовые движения BTC/ETH/SOL больше порога, 2021-2026.
     Правило «уже в цене» (движение за 7 дней до шока) на сотнях наблюдений, с разделением выборки по времени.
  B. FOMC глубже: типы заседаний (с прогнозами SEP / без, повышение / снижение / пауза), заявление vs пресс-конференция,
     протоколы FOMC (через 3 недели), речи в Джексон-Хоуле.
  C. Межрыночные связи на дневных данных: нефть (Brent, WTI), VIX, S&P 500 → BTC с лагом, и обратно.
  D. Внутридневная и недельная сезонность BTC как контекст для времени входа/выхода.

Запуск: python3 research/phase4.py --btc DIR --eth DIR --sol DIR --xasset DIR --out research/results
"""
import argparse, os
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy import stats
from phase2 import Px
from event_study import FOMC_DATES

NY = ZoneInfo("America/New_York")
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
THR = {"BTCUSDT": 0.03, "ETHUSDT": 0.04, "SOLUSDT": 0.06}

# Решения FOMC: изменение ставки в б.п. (0 = пауза). SEP-заседания: март, июнь, сентябрь, декабрь.
FOMC_CHANGE = {
    "2022-03-16": 25, "2022-05-04": 50, "2022-06-15": 75, "2022-07-27": 75, "2022-09-21": 75, "2022-11-02": 75, "2022-12-14": 50,
    "2023-02-01": 25, "2023-03-22": 25, "2023-05-03": 25, "2023-07-26": 25,
    "2024-09-18": -50, "2024-11-07": -25, "2024-12-18": -25,
    "2025-09-17": -25, "2025-10-29": -25, "2025-12-10": -25,
    "2026-09-16": 25,
}
JACKSON_HOLE = ["2021-08-27", "2022-08-26", "2023-08-25", "2024-08-23", "2025-08-22", "2026-08-28"]


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return (stats.ttest_1samp(x, 0).statistic if len(x) >= 3 else np.nan), len(x)


def cell(g):
    g = np.asarray(g, float); g = g[~np.isnan(g)]
    if len(g) < 5:
        return f"N={len(g)}"
    t, n = tstat(g)
    return f"{g.mean()*100:+.2f} / {t:+.1f} / {(g>0).mean()*100:.0f}% (N={n})"


# ------------------------------------------------------------------------------------------ A
def shock_universe(px: Px, thr):
    """Все моменты, когда |доходность за 60 мин| впервые превысила порог, не чаще раза в 48 часов."""
    c = px.c
    r60 = np.full(len(c), np.nan)
    r60[60:] = c[60:] / c[:-60] - 1
    idx = np.where(np.abs(r60) >= thr)[0]
    out = []; last = -10**9
    for i in idx:
        if px.ts[i] - last < 48 * 60:
            continue
        last = px.ts[i]
        t = pd.Timestamp(px.ts[i] * 60, unit="s", tz="UTC")   # конец часа, в котором порог пройден
        d = np.sign(r60[i])
        rec = {"t": t, "d": d, "r60": r60[i]}
        t_entry = t + pd.Timedelta(minutes=60)
        rec["pre7d"] = px.ret(t, -60 - 10080, -60)              # 7 дней до начала шокового часа
        rec["pre24h"] = px.ret(t, -60 - 1440, -60)
        rec["r_next60"] = px.ret(t, 0, 60)                       # час после шокового часа
        for k, lab in [(240, "4h"), (1440, "1d"), (4320, "3d"), (10080, "7d")]:
            rec[f"e_{lab}"] = px.ret(t_entry, 0, k)              # от входа (t+60) на k минут
        # объём шокового часа относительно среднего часового за 7 дней
        v = px.vol(t, -60, 0); base = px.vol(t, -60 - 10080, -60) / 168.0
        rec["volx"] = v / base if base > 0 else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def section_A(P, px_by_sym):
    P("# A. Вселенная шоков без ручного отбора (2021-2026)\n")
    P("Шок = первый момент, когда |доходность за 60 минут| превысила порог (BTC 3%, ETH 4%, SOL 6%), не чаще одного раза в 48 часов. "
      "Вход через 60 минут после конца шокового часа по направлению шока (continuation) или против (fade). "
      "«В цене» = движение за 7 дней до шокового часа в направлении шока (d × pre7d).\n")
    allU = {}
    for s in SYMS:
        U = shock_universe(px_by_sym[s], THR[s])
        U["aligned"] = U.d * U.pre7d
        U["year"] = U.t.dt.year
        allU[s] = U
        P(f"## {s}: шоков {len(U)}, из них вверх {int((U.d>0).sum())}, вниз {int((U.d<0).sum())}; по годам: {U.year.value_counts().sort_index().to_dict()}\n")
        P("Продолжение (d × доходность от входа t+60): средн. % / t / hit\n")
        P("| Группа | N | +4ч | +1д | +3д | +7д |")
        P("|---|---|---|---|---|---|")
        groups = [("все шоки", U.aligned.notna()),
                  ("в цене: pre7d в сторону шока > +5%", U.aligned > 0.05),
                  ("pre7d в сторону шока 0..5%", (U.aligned > 0) & (U.aligned <= 0.05)),
                  ("pre7d против шока 0..−5%", (U.aligned <= 0) & (U.aligned > -0.05)),
                  ("сюрприз: pre7d против шока < −5%", U.aligned <= -0.05),
                  ("шоки вверх", U.d > 0), ("шоки вниз", U.d < 0),
                  ("шоки вниз, в цене (> +5%)", (U.d < 0) & (U.aligned > 0.05)),
                  ("шоки вниз, сюрприз (< −5%)", (U.d < 0) & (U.aligned <= -0.05)),
                  ("шоки вверх, в цене (> +5%)", (U.d > 0) & (U.aligned > 0.05)),
                  ("шоки вверх, сюрприз (< −5%)", (U.d > 0) & (U.aligned <= -0.05)),
                  ("объём шокового часа > 5x", U.volx > 5),
                  ("час после шока продолжил движение", U.d * U.r_next60 > 0),
                  ("час после шока развернулся", U.d * U.r_next60 < 0)]
        for lab, m in groups:
            g = U[m]
            P(f"| {lab} | {len(g)} | " + " | ".join(cell(g.d * g[f"e_{h}"]) for h in ["4h", "1d", "3d", "7d"]) + " |")
        P("")
        # split sample
        P("Разделение выборки (вход t+60, выход +1д и +7д; continuation):\n")
        P("| Группа | 2021-2023: +1д | 2021-2023: +7д | 2024-2026: +1д | 2024-2026: +7д |")
        P("|---|---|---|---|---|")
        for lab, m in [("все шоки", U.aligned.notna()), ("в цене (> +5%)", U.aligned > 0.05), ("сюрприз (< −5%)", U.aligned <= -0.05),
                       ("вниз, в цене", (U.d < 0) & (U.aligned > 0.05)), ("вниз, сюрприз", (U.d < 0) & (U.aligned <= -0.05))]:
            a = U[m & (U.year <= 2023)]; b = U[m & (U.year >= 2024)]
            P(f"| {lab} | {cell(a.d*a.e_1d)} | {cell(a.d*a.e_7d)} | {cell(b.d*b.e_1d)} | {cell(b.d*b.e_7d)} |")
        P("")
        U.to_csv(os.path.join(OUT, f"shocks_{s}.csv"), index=False)
    # Portfolio-level: fade "в цене" shocks down vs continuation "сюрприз" — summary across symbols
    P("## Сводка по трём инструментам: лучшие и худшие ячейки (вход t+60, выход +1д)\n")
    rows = []
    for s, U in allU.items():
        for lab, m in [("в цене >+5%, fade", U.aligned > 0.05), ("сюрприз <−5%, continuation", U.aligned <= -0.05)]:
            g = U[m]; sign = -1 if "fade" in lab else 1
            x = (sign * g.d * g.e_1d).dropna(); t, n = tstat(x)
            rows.append((s, lab, n, x.mean() * 100, t, (x > 0).mean() * 100))
    P("| Инструмент | Правило | N | средн. % | t | hit |")
    P("|---|---|---|---|---|---|")
    for r in rows:
        P(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:+.2f} | {r[4]:+.1f} | {r[5]:.0f}% |")
    P("")


# ------------------------------------------------------------------------------------------ B
def fomc_times():
    rows = []
    for d in FOMC_DATES:
        t = pd.Timestamp(d + " 14:00", tz=NY).tz_convert("UTC")
        if t < pd.Timestamp("2021-01-01", tz="UTC") or t > pd.Timestamp("2026-10-20", tz="UTC"):
            continue
        m = int(d[5:7])
        rows.append({"d": d, "t": t, "sep": m in (3, 6, 9, 12), "chg": FOMC_CHANGE.get(d, 0)})
    return pd.DataFrame(rows)


def section_B(P, px_by_sym):
    P("# B. FOMC глубже\n")
    F = fomc_times()
    for s in SYMS:
        px = px_by_sym[s]
        R = F.copy()
        R["r_stmt"] = [px.ret(t, 0, 30) for t in R.t]        # заявление: 14:00-14:30
        R["r_press"] = [px.ret(t, 30, 90) for t in R.t]      # пресс-конференция: 14:30-15:30
        R["r60"] = [px.ret(t, 0, 60) for t in R.t]
        R["r_1d"] = [px.ret(t, 60, 1440) for t in R.t]       # от входа t+60 до +24ч
        R["r_2d"] = [px.ret(t, 60, 2880) for t in R.t]
        R["r_pre24"] = [px.ret(t, -1440, 0) for t in R.t]
        d = np.sign(R.r60)
        P(f"## {s}\n")
        P("| Разрез | N | K3 (по знаку 1-го часа, t+60 → +24ч): средн. % / t / hit | то же до +48ч |")
        P("|---|---|---|---|")
        for lab, m in [("все заседания", R.r60.notna()), ("с прогнозами SEP (мар/июн/сен/дек)", R.sep), ("без SEP", ~R.sep),
                       ("повышение ставки", R.chg > 0), ("снижение ставки", R.chg < 0), ("пауза", R.chg == 0),
                       ("|r60| > 1%", R.r60.abs() > 0.01), ("|r60| < 1%", R.r60.abs() <= 0.01),
                       ("заявление и пресс-конф. в одну сторону", np.sign(R.r_stmt) == np.sign(R.r_press)),
                       ("пресс-конф. развернула заявление", np.sign(R.r_stmt) != np.sign(R.r_press)),
                       ("1-й час против движения суток до", np.sign(R.r60) != np.sign(R.r_pre24)),
                       ("1-й час по движению суток до", np.sign(R.r60) == np.sign(R.r_pre24))]:
            g = R[m]; dd = np.sign(g.r60)
            P(f"| {lab} | {len(g)} | {cell(dd*g.r_1d)} | {cell(dd*g.r_2d)} |")
        P("")
        # что лучше предсказывает сутки: заявление или пресс-конференция
        P("Что предсказывает следующие 24 часа (Spearman с доходностью t+90 → +24ч):\n")
        r_after = np.array([px.ret(t, 90, 1440) for t in R.t])
        for lab, x in [("движение на заявлении (14:00-14:30)", R.r_stmt), ("движение на пресс-конференции (14:30-15:30)", R.r_press), ("весь первый час", R.r60)]:
            sp = stats.spearmanr(x, r_after, nan_policy="omit")
            P(f"- {lab}: rho={sp.statistic:+.2f}, p={sp.pvalue:.2f}")
        P("")

    # Протоколы FOMC (через 21 день, 14:00 ET) и Джексон-Хоул (10:00 ET)
    P("## Протоколы FOMC и Джексон-Хоул: работает ли то же правило\n")
    P("| Событие | Инструмент | N | медиана \\|r60\\| | контроль \\|r60\\| | K3-подобное t+60 → +24ч: средн. % / t / hit |")
    P("|---|---|---|---|---|---|")
    minutes = [pd.Timestamp(d + " 14:00", tz=NY) + pd.Timedelta(days=21) for d in FOMC_DATES]
    minutes = [t.tz_convert("UTC") for t in minutes if pd.Timestamp("2021-01-01", tz="UTC") <= t.tz_convert("UTC") <= pd.Timestamp("2026-10-20", tz="UTC")]
    jh = [pd.Timestamp(d + " 10:00", tz=NY).tz_convert("UTC") for d in JACKSON_HOLE]
    for lab, T in [("Протоколы FOMC", minutes), ("Джексон-Хоул", jh)]:
        for s in SYMS:
            px = px_by_sym[s]
            r60 = np.array([px.ret(t, 0, 60) for t in T]); r1d = np.array([px.ret(t, 60, 1440) for t in T])
            ctl = np.array([px.ret(t - pd.Timedelta(days=1), 0, 60) for t in T])
            P(f"| {lab} | {s} | {len(T)} | {np.nanmedian(np.abs(r60))*100:.2f}% | {np.nanmedian(np.abs(ctl))*100:.2f}% | {cell(np.sign(r60)*r1d)} |")
    P("")


# ------------------------------------------------------------------------------------------ C
def daily_btc_at_ny_close(px: Px, dates):
    """Цена BTC в 16:00 Нью-Йорка для каждой даты."""
    out = {}
    for d in dates:
        t = pd.Timestamp(str(d) + " 16:00", tz=NY).tz_convert("UTC")
        out[d] = px.px(t)
    return pd.Series(out)


def section_C(P, px_by_sym, xdir):
    P("# C. Межрыночные связи на дневных данных (выравнивание по закрытию Нью-Йорка 16:00 ET)\n")
    brent = pd.read_csv(os.path.join(xdir, "brent.csv"), parse_dates=["Date"]).set_index("Date")["Price"]
    wti = pd.read_csv(os.path.join(xdir, "wti.csv"), parse_dates=["Date"]).set_index("Date")["Price"]
    vix = pd.read_csv(os.path.join(xdir, "vix.csv"), parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
    spx = pd.read_csv(os.path.join(xdir, "us_market.csv"), parse_dates=["Date"]).set_index("Date")["Close"]
    X = pd.concat({"brent": brent, "wti": wti, "vix": vix, "spx": spx}, axis=1)
    X = X[(X.index >= "2021-01-01")]
    for s in ["BTCUSDT", "ETHUSDT"]:
        px = px_by_sym[s]
        dates = X.index[X.index <= pd.Timestamp("2026-10-20")]
        b = daily_btc_at_ny_close(px, dates.date)
        b.index = pd.to_datetime(b.index)
        D = X.join(b.rename("btc"), how="inner").dropna(subset=["btc"])
        R = pd.DataFrame({
            "btc": D.btc.pct_change(), "brent": D.brent.pct_change(), "wti": D.wti.pct_change(),
            "dvix": D.vix.pct_change(), "spx": D.spx.pct_change(),
        }).dropna(how="all")
        P(f"## {s} ({len(R)} торговых дней)\n")
        P("Spearman между переменной в день t и доходностью крипты в день t (same day), t+1 (lead 1), t+1..t+3 (lead 3):\n")
        P("| Переменная | same day | lead 1 | lead 3 | N |")
        P("|---|---|---|---|---|")
        for v in ["brent", "wti", "dvix", "spx"]:
            x = R[v]; y0 = R.btc; y1 = R.btc.shift(-1); y3 = R.btc.shift(-1).rolling(3).sum().shift(-2)
            m = x.notna()
            s0 = stats.spearmanr(x[m], y0[m], nan_policy="omit"); s1 = stats.spearmanr(x[m], y1[m], nan_policy="omit"); s3 = stats.spearmanr(x[m], y3[m], nan_policy="omit")
            P(f"| {v} | {s0.statistic:+.3f} (p={s0.pvalue:.2f}) | {s1.statistic:+.3f} (p={s1.pvalue:.2f}) | {s3.statistic:+.3f} (p={s3.pvalue:.2f}) | {int(m.sum())} |")
        P("")
        P("Событийно: крупные дневные движения других рынков → крипта в следующие дни (средн. % / t / hit):\n")
        P("| Условие | N | крипта same day | крипта +1д | крипта +1..+3д | крипта +1..+5д |")
        P("|---|---|---|---|---|---|")
        nxt1 = R.btc.shift(-1); nxt3 = R.btc.shift(-1).rolling(3).sum().shift(-2); nxt5 = R.btc.shift(-1).rolling(5).sum().shift(-4)
        for lab, m in [("Brent > +4%", R.brent > 0.04), ("Brent < −4%", R.brent < -0.04), ("VIX > +15%", R.dvix > 0.15), ("VIX < −10%", R.dvix < -0.10),
                       ("S&P < −2%", R.spx < -0.02), ("S&P > +2%", R.spx > 0.02), ("S&P < −1% и Brent > +3% (стагфляционный день)", (R.spx < -0.01) & (R.brent > 0.03))]:
            P(f"| {lab} | {int(m.sum())} | {cell(R.btc[m])} | {cell(nxt1[m])} | {cell(nxt3[m])} | {cell(nxt5[m])} |")
        P("")
        # does crypto lead equities? weekend BTC return -> Monday SPX
        P("Ведёт ли крипта: доходность BTC за выходные (пт 16:00 → пн 09:30 ET) и S&P в понедельник:\n")
        mon = [d for d in R.index if d.weekday() == 0]
        rows = []
        for d in mon:
            fri = d - pd.Timedelta(days=3)
            p_fri = px.px(pd.Timestamp(str(fri.date()) + " 16:00", tz=NY).tz_convert("UTC"))
            p_mon = px.px(pd.Timestamp(str(d.date()) + " 09:30", tz=NY).tz_convert("UTC"))
            if np.isnan(p_fri) or np.isnan(p_mon) or d not in R.index or pd.isna(R.spx.get(d)):
                continue
            rows.append((p_mon / p_fri - 1, R.spx[d]))
        W = pd.DataFrame(rows, columns=["btc_wknd", "spx_mon"])
        sp = stats.spearmanr(W.btc_wknd, W.spx_mon)
        P(f"- N={len(W)}, Spearman={sp.statistic:+.3f}, p={sp.pvalue:.3f}; S&P в понедельник после BTC-выходных < −3%: {cell(W.spx_mon[W.btc_wknd < -0.03])}; после > +3%: {cell(W.spx_mon[W.btc_wknd > 0.03])}")
        P("")


# ------------------------------------------------------------------------------------------ D
def section_D(P, px):
    P("# D. Сезонность BTC по часам UTC и дням недели (2021-2026), средняя доходность за час в б.п.\n")
    c = pd.Series(px.c, index=pd.to_datetime(px.ts * 60, unit="s", utc=True))
    h = c.resample("1h").last().dropna()
    r = h.pct_change().dropna() * 1e4
    tab = r.groupby([r.index.dayofweek, r.index.hour]).mean().unstack()
    tab.index = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    P("Средняя доходность по часам (столбцы = час UTC начала интервала), б.п.:\n")
    P("| День | " + " | ".join(f"{hh:02d}" for hh in range(24)) + " |")
    P("|---|" + "---|" * 24)
    for d in tab.index:
        P(f"| {d} | " + " | ".join(f"{tab.loc[d, hh]:+.1f}" for hh in range(24)) + " |")
    P("")
    byh = r.groupby(r.index.hour).agg(["mean", "count"])
    byh["t"] = byh["mean"] / (r.groupby(r.index.hour).std() / np.sqrt(byh["count"]))
    top = byh.sort_values("mean")
    P(f"- Худшие часы UTC (средн. б.п., t): " + ", ".join(f"{hh:02d}: {row['mean']:+.1f} (t={row['t']:+.1f})" for hh, row in top.head(4).iterrows()))
    P(f"- Лучшие часы UTC: " + ", ".join(f"{hh:02d}: {row['mean']:+.1f} (t={row['t']:+.1f})" for hh, row in top.tail(4).iterrows()))
    byd = r.groupby(r.index.dayofweek).sum() / (len(r) / 24 / 7)
    P(f"- Средняя дневная доходность по дням недели (б.п.): " + ", ".join(f"{n}: {v:+.0f}" for n, v in zip(["Пн","Вт","Ср","Чт","Пт","Сб","Вс"], byd.values)))
    P("")


OUT = "research/results"

def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc", required=True); ap.add_argument("--eth", required=True); ap.add_argument("--sol", required=True)
    ap.add_argument("--xasset", required=True); ap.add_argument("--out", default="research/results")
    a = ap.parse_args(); OUT = a.out; os.makedirs(OUT, exist_ok=True)
    px = {"BTCUSDT": Px(a.btc, "BTCUSDT"), "ETHUSDT": Px(a.eth, "ETHUSDT"), "SOLUSDT": Px(a.sol, "SOLUSDT")}
    L = []; P = L.append
    section_A(P, px)
    section_B(P, px)
    section_C(P, px, a.xasset)
    section_D(P, px["BTCUSDT"])
    txt = "\n".join(L)
    open(os.path.join(OUT, "results_phase4.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
