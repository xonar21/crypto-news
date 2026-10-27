"""
Событийное исследование: реакция BTCUSDT (Binance spot, 1m) на макроновости США.

Данные:
  - свечи 1m: parshant231234-eng/BTCUSDT-Historical-Data (GitHub raw)
  - календарь релизов с консенсусом Investing.com: superpilot69/fred-us-macro-open-data
  - даты FOMC 2021-2025: захардкожены (statement 14:00 ET)

Запуск:  python3 research/event_study.py --data DIR_WITH_CSV --events events.json --out research/results
"""
import argparse, glob, json, os
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy import stats

HORIZONS = [1, 5, 15, 30, 60, 240, 1440]          # минуты после релиза
FEE_RT = 0.0010                                   # 0.10% round-trip (фьючерсы Binance, тейкер 0.05% x2)
SLIP = 0.0005                                     # 0.05% проскальзывание на входе+выходе суммарно (консервативно)
COST = FEE_RT + SLIP

# Ожидаемый знак реакции BTC на положительный сюрприз (actual > forecast).
# -1: сильнее ожиданий = ястребино/risk-off = BTC вниз.
SIGN = {
    "CPIAUCNS": -1, "CPILFENS": -1, "PPIACO": -1, "PCEPI": -1, "PCEPILFE": -1,
    "PAYEMS": -1, "CES0500000003": -1, "ADPMNUSNERSA": -1, "JTSJOL": -1, "RSAFS": -1, "GDPC1": -1,
    "UNRATE": +1,      # безработица выше ожиданий = голубино = BTC вверх
    "ICSA": +1,        # заявки выше = слабый рынок труда = голубино
}
NAMES = {
    "CPIAUCNS": "CPI YoY", "CPILFENS": "Core CPI YoY", "PPIACO": "PPI YoY", "PCEPI": "PCE YoY",
    "PCEPILFE": "Core PCE YoY", "PAYEMS": "NFP", "CES0500000003": "Avg hourly earnings",
    "ADPMNUSNERSA": "ADP", "JTSJOL": "JOLTS", "RSAFS": "Retail sales", "GDPC1": "Real GDP",
    "UNRATE": "Unemployment", "ICSA": "Initial claims", "FOMC": "FOMC statement",
}

FOMC_DATES = [  # день публикации заявления, 14:00 America/New_York
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16",
]


def load_prices(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "BTCUSDT_1m_*.csv")))
    df = pd.concat([pd.read_csv(f, usecols=["timestamp", "open", "high", "low", "close", "volume"]) for f in files])
    df["t"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("t").set_index("t").sort_index()
    return df


def load_events(path):
    ev = json.load(open(path))["events"]
    rows = []
    for e in ev:
        m = e["metadata"]
        if m.get("releaseDateApproximate"):
            continue
        cons = m.get("consensus") or {}
        if cons.get("forecast") is None or cons.get("actual") is None:
            continue
        sid = m["seriesId"]
        if sid not in SIGN:
            continue
        rows.append({
            "series": sid, "name": NAMES[sid], "t": pd.Timestamp(e["createdAt"]),
            "actual": cons["actual"], "forecast": cons["forecast"], "surprise": cons["actual"] - cons["forecast"],
        })
    df = pd.DataFrame(rows)
    ny = ZoneInfo("America/New_York")
    f = pd.DataFrame({
        "series": "FOMC", "name": "FOMC statement",
        "t": [pd.Timestamp(d + " 14:00", tz=ny).tz_convert("UTC") for d in FOMC_DATES],
        "actual": np.nan, "forecast": np.nan, "surprise": np.nan,
    })
    df = pd.concat([df, f], ignore_index=True)
    df = df[df["t"] >= "2021-01-01"].sort_values("t").reset_index(drop=True)
    # стандартизируем сюрприз внутри серии (в единицах исторического std сюрприза)
    df["z"] = df.groupby("series")["surprise"].transform(lambda s: s / s.std(ddof=0) if s.std(ddof=0) > 0 else np.nan)
    return df


def px_at(close, open_, t):
    """Цена 'в момент t' = open свечи, открывающейся в t (если нет свечи - ближайшая следующая)."""
    i = open_.index.searchsorted(t)
    if i >= len(open_):
        return np.nan
    return open_.iloc[i]


def event_returns(prices, events):
    o, c, h, l = prices["open"], prices["close"], prices["high"], prices["low"]
    out = []
    for _, e in events.iterrows():
        t0 = e["t"]
        p0 = px_at(c, o, t0)
        if np.isnan(p0):
            continue
        rec = e.to_dict()
        for k in HORIZONS:
            rec[f"r{k}"] = px_at(c, o, t0 + pd.Timedelta(minutes=k)) / p0 - 1
        # контрольное окно: тот же интервал за сутки до релиза
        for k in [15, 60]:
            pc = px_at(c, o, t0 - pd.Timedelta(days=1))
            rec[f"ctl_r{k}"] = px_at(c, o, t0 - pd.Timedelta(days=1) + pd.Timedelta(minutes=k)) / pc - 1
        # realized vol: сумма |1m ret| за 60 минут после vs за 60 минут до
        post = c.loc[t0: t0 + pd.Timedelta(minutes=60)]
        pre = c.loc[t0 - pd.Timedelta(minutes=61): t0 - pd.Timedelta(minutes=1)]
        rec["rv_post"] = np.log(post).diff().abs().sum()
        rec["rv_pre"] = np.log(pre).diff().abs().sum()
        rec["range60"] = (h.loc[t0: t0 + pd.Timedelta(minutes=59)].max() / l.loc[t0: t0 + pd.Timedelta(minutes=59)].min() - 1)
        out.append(rec)
    return pd.DataFrame(out)


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    t, p = stats.ttest_1samp(x, 0)
    return t, p, len(x)


def summarize(er, out_dir):
    lines = []
    P = lines.append

    # 1. Движется ли цена вообще: |r| после релиза vs контроль
    os.makedirs(out_dir, exist_ok=True)
    P("## 1. Есть ли реакция вообще (абсолютное движение, медиана, %)\n")
    P("| Релиз | N | |r15| событие | |r15| контроль | |r60| событие | |r60| контроль | RV60 post/pre | диапазон 60м |")
    P("|---|---|---|---|---|---|---|---|")
    for s, g in er.groupby("name"):
        P(f"| {s} | {len(g)} | {g.r15.abs().median()*100:.3f} | {g.ctl_r15.abs().median()*100:.3f} | "
          f"{g.r60.abs().median()*100:.3f} | {g.ctl_r60.abs().median()*100:.3f} | "
          f"{(g.rv_post/g.rv_pre).median():.2f} | {g.range60.median()*100:.2f} |")
    g = er
    P(f"| **все события** | {len(g)} | {g.r15.abs().median()*100:.3f} | {g.ctl_r15.abs().median()*100:.3f} | "
      f"{g.r60.abs().median()*100:.3f} | {g.ctl_r60.abs().median()*100:.3f} | {(g.rv_post/g.rv_pre).median():.2f} | {g.range60.median()*100:.2f} |")
    mw = stats.mannwhitneyu(er.r60.abs().dropna(), er.ctl_r60.abs().dropna(), alternative="greater")
    P(f"\nMann-Whitney |r60| событие > контроль: p = {mw.pvalue:.2e}\n")

    # 2. Направление: корреляция сюрприза и доходности
    P("## 2. Предсказывает ли знак сюрприза направление (surprise -> return)\n")
    P("Ожидаемый знак: сильнее-ожиданий по инфляции/занятости = BTC вниз; безработица/заявки выше = BTC вверх.\n")
    P("| Релиз | N | Spearman r5 | Spearman r15 | Spearman r60 | Spearman r240 | hit-rate r15 | hit-rate r60 | p(hit60) |")
    P("|---|---|---|---|---|---|---|---|---|")
    with_s = er.dropna(subset=["z"])
    with_s = with_s[with_s.surprise != 0]
    def row(name, g):
        sgn = g.series.map(SIGN).values
        cells = []
        for k in [5, 15, 60, 240]:
            r = stats.spearmanr(g.z.values * sgn, g[f"r{k}"].values, nan_policy="omit")
            cells.append(f"{r.statistic:+.2f} (p={r.pvalue:.2f})")
        hits15 = np.mean(np.sign(g.r15.values) == sgn * np.sign(g.surprise.values))
        hit60 = (np.sign(g.r60.values) == sgn * np.sign(g.surprise.values))
        pb = stats.binomtest(int(hit60.sum()), len(hit60), 0.5, alternative="greater").pvalue
        P(f"| {name} | {len(g)} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {hits15*100:.0f}% | {hit60.mean()*100:.0f}% | {pb:.2f} |")
    for s, g in with_s.groupby("name"):
        if len(g) >= 15:
            row(s, g)
    row("**все с сюрпризом**", with_s)
    infl = with_s[with_s.series.isin(["CPIAUCNS", "CPILFENS"])]
    row("**CPI + Core CPI**", infl)
    P("")

    # --- группировка одновременных релизов (CPI+Core CPI, NFP+UNRATE+AHE и т.д.) в одно событие ---
    def composite(g):
        sgn = g.series.map(SIGN)
        zz = (g.z * sgn).dropna()
        first = g.iloc[0]
        rec = {c: first[c] for c in er.columns if c.startswith("r") or c.startswith("ctl_") or c in ("rv_post", "rv_pre", "range60", "t")}
        rec["labels"] = "+".join(sorted(set(g["name"])))
        rec["zc"] = zz.mean() if len(zz) else np.nan           # композитный сюрприз, уже со знаком "ожидаемого направления BTC"
        rec["is_cpi"] = bool(g.series.isin(["CPIAUCNS", "CPILFENS"]).any())
        rec["is_nfp"] = bool(g.series.isin(["PAYEMS", "UNRATE"]).any())
        rec["is_fomc"] = bool((g.series == "FOMC").any())
        return pd.Series(rec)
    ge = er.assign(_t=er.t).groupby("_t").apply(composite).reset_index(drop=True)
    ge["t"] = pd.to_datetime(ge["t"], utc=True)
    P(f"Уникальных моментов выхода новостей (после объединения одновременных релизов): {len(ge)}, "
      f"из них CPI: {ge.is_cpi.sum()}, NFP: {ge.is_nfp.sum()}, FOMC: {ge.is_fomc.sum()}\n")

    COSTS = [0.0005, 0.0010, 0.0015]   # оптимист (мейкер/VIP), база, консервативно
    HK = [15, 30, 60, 240, 1440]

    def strat_table(name, g, direction, hdr_done=[False]):
        """direction: массив +1/-1 на событие; вход по цене t+1, выход t+k."""
        for k in HK:
            gross = direction * ((1 + g[f"r{k}"].values) / (1 + g.r1.values) - 1)
            t, p, n = tstat(gross)
            nets = [np.nanmean(gross) - c for c in COSTS]
            P(f"| {name} | {n} | {k}м | {np.nanmean(gross)*100:+.3f} | {np.nanmedian(gross)*100:+.3f} | {t:+.2f} | "
              f"{np.nanmean(gross>0)*100:.0f}% | {nets[0]*100:+.3f} | {nets[1]*100:+.3f} | {nets[2]*100:+.3f} | "
              f"{np.nanmean(gross)/np.nanstd(gross):+.2f} |")

    HDR = ("| Выборка | N | выход | средн. gross % | медиана % | t-stat | hit-rate | net @0.05% | net @0.10% | net @0.15% | Sharpe/сделку |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|")

    # 3. Стратегия A: по знаку сюрприза
    P("## 3. Стратегия A: торговля по знаку сюрприза (вход через 1 мин после релиза, |z композитный| >= 0.5)\n")
    P("Доходность gross на сделку; net = gross минус издержки round-trip (комиссии + проскальзывание) в трёх сценариях.\n")
    P(HDR)
    ga = ge[ge.zc.abs() >= 0.5]
    strat_table("все релизы", ga, np.sign(ga.zc.values))
    strat_table("CPI", ga[ga.is_cpi], np.sign(ga[ga.is_cpi].zc.values))
    strat_table("NFP/безработица", ga[ga.is_nfp], np.sign(ga[ga.is_nfp].zc.values))
    strat_table("прочие релизы", ga[~ga.is_cpi & ~ga.is_nfp], np.sign(ga[~ga.is_cpi & ~ga.is_nfp].zc.values))
    P("")
    P("### 3b. Стратегия A по годам (все релизы, |z|>=0.5)\n")
    P(HDR)
    for yr, g in ga.groupby(ga.t.dt.year):
        if len(g) >= 10:
            strat_table(f"{yr}", g, np.sign(g.zc.values))
    P("")
    P("### 3c. Стратегия A только CPI по годам\n")
    P(HDR)
    for yr, g in ga[ga.is_cpi].groupby(ga[ga.is_cpi].t.dt.year):
        if len(g) >= 5:
            strat_table(f"CPI {yr}", g, np.sign(g.zc.values))
    P("")

    # 4. Стратегия B: продолжение первого импульса (без знания сюрприза)
    P("## 4. Стратегия B: продолжение импульса (вход по знаку первой 1-мин свечи после релиза, выход через k минут)\n")
    P(HDR)
    def sb(name, g, thresh=0.0):
        g = g[g.r1.abs() > thresh]
        if len(g) >= 10:
            strat_table(name, g, np.sign(g.r1.values))
    sb("все события", ge)
    sb("все события, |r1|>0.3%", ge, 0.003)
    sb("CPI", ge[ge.is_cpi])
    sb("CPI, |r1|>0.3%", ge[ge.is_cpi], 0.003)
    sb("FOMC", ge[ge.is_fomc])
    sb("NFP", ge[ge.is_nfp])
    P("")
    P("### 4b. Стратегия B, CPI+FOMC по годам\n")
    P(HDR)
    gcf = ge[ge.is_cpi | ge.is_fomc]
    for yr, g in gcf.groupby(gcf.t.dt.year):
        if len(g) >= 5:
            strat_table(f"CPI+FOMC {yr}", g, np.sign(g.r1.values))
    P("")

    # 4c. Стратегия C: разворот (fade) — вход в t+60 против первого импульса, выход t+240 / t+1440
    P("## 4c. Стратегия C: разворот. Вход в t+60 ПРОТИВ направления первого часа, выход через 240 и 1440 мин\n")
    P("| Выборка | N | выход | средн. gross % | медиана % | t-stat | hit-rate | net @0.10% |")
    P("|---|---|---|---|---|---|---|---|")
    def sc(name, g, thresh):
        g = g[g.r60.abs() > thresh]
        if len(g) < 10:
            return
        d = -np.sign(g.r60.values)
        for k in [240, 1440]:
            gross = d * ((1 + g[f"r{k}"].values) / (1 + g.r60.values) - 1)
            t, p, n = tstat(gross)
            P(f"| {name} | {n} | {k}м | {np.nanmean(gross)*100:+.3f} | {np.nanmedian(gross)*100:+.3f} | {t:+.2f} | "
              f"{np.nanmean(gross>0)*100:.0f}% | {(np.nanmean(gross)-0.001)*100:+.3f} |")
    sc("все события, |r60|>0.5%", ge, 0.005)
    sc("все события, |r60|>1%", ge, 0.01)
    sc("CPI, |r60|>0.5%", ge[ge.is_cpi], 0.005)
    sc("FOMC, |r60|>0.5%", ge[ge.is_fomc], 0.005)
    P("")

    # 5. Сколько движения происходит в первые минуты
    P("## 5. Какая доля часового движения реализуется в первые минуты (уникальные события с |r60| > 0.5%)\n")
    big = ge[ge.r60.abs() > 0.005]
    for k in [1, 5, 15, 30]:
        share = (big[f"r{k}"] / big.r60).clip(-2, 2)
        P(f"- r{k}/r60: медиана {share.median()*100:.0f}%, доля событий где знак r{k} совпал со знаком r60: {(np.sign(big[f'r{k}'])==np.sign(big.r60)).mean()*100:.0f}% (N={len(big)})")
    P("")
    ge.to_csv(os.path.join(out_dir, "event_returns_grouped.csv"), index=False)

    # 6. По годам: чувствительность CPI
    P("## 6. CPI + Core CPI по годам: Spearman(сюрприз со знаком, r60) и медиана |r60|\n")
    P("| Год | N | Spearman r60 | p | медиана |r60| % | медиана |r60| контроль % |")
    P("|---|---|---|---|---|---|")
    for yr, g in infl.groupby(infl.t.dt.year):
        sgn = g.series.map(SIGN).values
        r = stats.spearmanr(g.z.values * sgn, g.r60.values, nan_policy="omit")
        P(f"| {yr} | {len(g)} | {r.statistic:+.2f} | {r.pvalue:.2f} | {g.r60.abs().median()*100:.3f} | {g.ctl_r60.abs().median()*100:.3f} |")
    P("")

    txt = "\n".join(lines)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "results.md"), "w").write(txt)
    er.to_csv(os.path.join(out_dir, "event_returns.csv"), index=False)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", default="research/results")
    a = ap.parse_args()
    prices = load_prices(a.data)
    print("prices:", prices.index.min(), prices.index.max(), len(prices))
    events = load_events(a.events)
    print("events:", len(events), events.groupby("name").size().to_dict())
    er = event_returns(prices, events)
    summarize(er, a.out)
