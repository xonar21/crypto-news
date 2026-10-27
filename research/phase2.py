"""
Фаза 2: расширение горизонтов и инструментов, стратегии без гонки за скорость.

  A. BTC / ETH / SOL: отложенные входы (t+5, t+15, t+30) после макрорелизов с техническими фильтрами
     (согласие импульса с сюрпризом, пробой пред-релизного диапазона, всплеск объёма, тренд).
  B. «Стрэддл» отложенными ордерами, выставленными ДО релиза (стоп-ордера на пробой диапазона).
  C. Дневные горизонты: дрейф до FOMC/CPI, дрейф после события, продолжение на следующий день.
  D. Внеплановые крипто-новости (CryptoPanic, 248k новостей с секундными метками, голоса пользователей).

Запуск:
  python3 research/phase2.py --btc DIR --eth DIR --sol DIR --events events.json --news joined.csv --out research/results
"""
import argparse, glob, json, os
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy import stats

from event_study import SIGN, NAMES, FOMC_DATES, load_events  # переиспользуем календарь

COST = 0.0010  # round-trip, базовый сценарий


class Px:
    """Минутные бары одного инструмента в numpy, быстрые выборки по времени."""

    def __init__(self, data_dir, sym):
        files = sorted(glob.glob(os.path.join(data_dir, f"{sym}_1m_*.csv")))
        df = pd.concat([pd.read_csv(f, usecols=["timestamp", "open", "high", "low", "close", "volume"]) for f in files])
        df = df.drop_duplicates("timestamp").sort_values("timestamp")
        self.sym = sym
        self.ts = df["timestamp"].to_numpy(np.int64) // 60000  # минуты unix
        self.o = df["open"].to_numpy(float); self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float); self.c = df["close"].to_numpy(float)
        self.v = df["volume"].to_numpy(float)

    def idx(self, t):
        m = int(pd.Timestamp(t).value // 60_000_000_000)
        return int(np.searchsorted(self.ts, m))

    def px(self, t):
        i = self.idx(t)
        return self.o[i] if i < len(self.o) else np.nan

    def ret(self, t0, k_from, k_to):
        a = self.px(t0 + pd.Timedelta(minutes=k_from)); b = self.px(t0 + pd.Timedelta(minutes=k_to))
        return b / a - 1

    def rng(self, t0, k_from, k_to):
        """max high / min low в окне [t0+k_from, t0+k_to)."""
        i, j = self.idx(t0 + pd.Timedelta(minutes=k_from)), self.idx(t0 + pd.Timedelta(minutes=k_to))
        if j <= i:
            return np.nan, np.nan
        return self.h[i:j].max(), self.l[i:j].min()

    def vol(self, t0, k_from, k_to):
        i, j = self.idx(t0 + pd.Timedelta(minutes=k_from)), self.idx(t0 + pd.Timedelta(minutes=k_to))
        return self.v[i:j].sum() if j > i else np.nan

    def path_exit(self, t0, k_entry, k_exit, direction, stop_px):
        """Вход по open в t0+k_entry, стоп по касанию high/low, иначе выход по open в t0+k_exit. Возвращает доходность."""
        i, j = self.idx(t0 + pd.Timedelta(minutes=k_entry)), self.idx(t0 + pd.Timedelta(minutes=k_exit))
        if j <= i or j >= len(self.o):
            return np.nan
        e = self.o[i]
        if direction > 0:
            hit = np.where(self.l[i:j] <= stop_px)[0]
            x = stop_px if len(hit) else self.o[j]
            return x / e - 1
        else:
            hit = np.where(self.h[i:j] >= stop_px)[0]
            x = stop_px if len(hit) else self.o[j]
            return e / x - 1


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, len(x)
    return stats.ttest_1samp(x, 0).statistic, len(x)


def row(P, name, gross, cost=COST):
    t, n = tstat(gross)
    g = np.asarray(gross, float); g = g[~np.isnan(g)]
    if n < 5:
        return
    P(f"| {name} | {n} | {g.mean()*100:+.3f} | {np.median(g)*100:+.3f} | {t:+.2f} | {(g>0).mean()*100:.0f}% | "
      f"{(g.mean()-cost)*100:+.3f} | {g.mean()/g.std():+.2f} |")


HDR = "| Вариант | N | gross % | медиана % | t | hit | net @0.10% | Sharpe/сделку |\n|---|---|---|---|---|---|---|---|"


# ----------------------------------------------------------------------------------------------------
def grouped_events(events):
    """Объединяем одновременные релизы, композитный сюрприз со знаком ожидаемого влияния на BTC."""
    rows = []
    for t, g in events.groupby("t"):
        zz = (g["z"] * g["series"].map(SIGN)).dropna()
        rows.append({"t": t, "labels": "+".join(sorted(set(g["name"]))), "zc": zz.mean() if len(zz) else np.nan,
                     "is_cpi": bool(g["series"].isin(["CPIAUCNS", "CPILFENS"]).any()),
                     "is_nfp": bool(g["series"].isin(["PAYEMS", "UNRATE"]).any()),
                     "is_fomc": bool((g["series"] == "FOMC").any())})
    return pd.DataFrame(rows)


def features(px: Px, ge):
    """Признаки и доходности для каждого события на одном инструменте."""
    out = []
    for _, e in ge.iterrows():
        t0 = e["t"]
        r = {"t": t0, "labels": e["labels"], "zc": e["zc"], "is_cpi": e["is_cpi"], "is_nfp": e["is_nfp"], "is_fomc": e["is_fomc"]}
        for k in [1, 5, 15, 30, 60, 240, 1440]:
            r[f"r{k}"] = px.ret(t0, 0, k)
        r["r_pre24"] = px.ret(t0, -1440, 0)          # тренд суток до релиза
        r["r_pre1h"] = px.ret(t0, -60, 0)
        hi, lo = px.rng(t0, -60, 0)                  # пред-релизный диапазон (60 мин)
        r["pre_hi"], r["pre_lo"] = hi, lo
        hi30, lo30 = px.rng(t0, -30, 0)
        r["pre_hi30"], r["pre_lo30"] = hi30, lo30
        p0 = px.px(t0)
        r["pre_range_pct"] = (hi - lo) / p0
        v5 = px.vol(t0, 0, 5)
        base = px.vol(t0, -1440, 0) / 288.0          # средний 5-мин объём за предыдущие сутки
        r["vol_ratio5"] = v5 / base if base > 0 else np.nan
        r["p0"] = p0
        for k in [5, 15, 30]:
            r[f"p{k}"] = px.px(t0 + pd.Timedelta(minutes=k))
        # контроль: тот же час накануне
        r["ctl_r60"] = px.ret(t0 - pd.Timedelta(days=1), 0, 60)
        r["ctl_r1440"] = px.ret(t0 - pd.Timedelta(days=1), 0, 1440)
        out.append(r)
    return pd.DataFrame(out)


def section_A(P, feats_by_sym):
    P("# A. Отложенные входы после макрорелизов с техническими фильтрами (BTC / ETH / SOL)\n")
    P("Вход по open свечи через k_entry минут после релиза, выход через k_exit минут после релиза. "
      "Направление: по знаку импульса r(0→k_entry). Издержки 0.10% round-trip.\n")
    for sym, F in feats_by_sym.items():
        P(f"## {sym}\n")
        # базовая реакция
        for grp, mask in [("CPI", F.is_cpi), ("FOMC", F.is_fomc), ("NFP", F.is_nfp), ("все релизы", F.is_cpi | ~F.is_cpi)]:
            g = F[mask]
            P(f"- {grp}: N={len(g)}, медиана |r60| {g.r60.abs().median()*100:.2f}% vs контроль {g.ctl_r60.abs().median()*100:.2f}%, "
              f"медиана |r1440| {g.r1440.abs().median()*100:.2f}% vs контроль {g.ctl_r1440.abs().median()*100:.2f}%")
        P("")
        for grp, mask in [("CPI", F.is_cpi), ("CPI+FOMC", F.is_cpi | F.is_fomc), ("все релизы", F.is_cpi | ~F.is_cpi)]:
            G = F[mask].copy()
            P(f"### {sym} — {grp} (N={len(G)})\n")
            P(HDR)
            for ke in [5, 15, 30]:
                rk = G[f"r{ke}"]
                d = np.sign(rk)
                for kx in [60, 240, 1440]:
                    if kx <= ke:
                        continue
                    gross = d * ((1 + G[f"r{kx}"]) / (1 + rk) - 1)
                    row(P, f"импульс, вход t+{ke}, выход t+{kx}", gross)
                    # фильтр 1: согласие импульса с сюрпризом
                    agree = (np.sign(G.zc) == d) & G.zc.notna() & (G.zc.abs() >= 0.3)
                    row(P, f"  + импульс согласен с сюрпризом (|z|>=0.3), вход t+{ke}, выход t+{kx}", gross[agree])
                    disagree = (np.sign(G.zc) == -d) & G.zc.notna() & (G.zc.abs() >= 0.3)
                    row(P, f"  + импульс ПРОТИВ сюрприза (|z|>=0.3), вход t+{ke}, выход t+{kx}", gross[disagree])
                    # фильтр 2: пробой пред-релизного диапазона (60 мин) к моменту входа
                    pk = G[f"p{ke}"]
                    brk = ((d > 0) & (pk > G.pre_hi)) | ((d < 0) & (pk < G.pre_lo))
                    row(P, f"  + цена пробила диапазон 60м до релиза, вход t+{ke}, выход t+{kx}", gross[brk])
                    # фильтр 3: объём первых 5 минут > 3x нормы
                    volf = G.vol_ratio5 > 3
                    row(P, f"  + объём 5м > 3x нормы, вход t+{ke}, выход t+{kx}", gross[volf])
                    # фильтр 4: тренд суток совпадает с импульсом
                    trend = np.sign(G.r_pre24) == d
                    row(P, f"  + тренд 24ч совпадает, вход t+{ke}, выход t+{kx}", gross[trend])
                    # комбинация: пробой + объём
                    row(P, f"  + пробой И объём>3x, вход t+{ke}, выход t+{kx}", gross[brk & volf])
                    # комбинация: пробой + согласие с сюрпризом
                    row(P, f"  + пробой И согласие с сюрпризом, вход t+{ke}, выход t+{kx}", gross[brk & agree])
            P("")
            # по годам для одного репрезентативного варианта: вход t+15, выход t+240, пробой
            P(f"Вариант «пробой диапазона, вход t+15, выход t+240» по годам ({sym}, {grp}):\n")
            P(HDR)
            d = np.sign(G.r15); brk = ((d > 0) & (G.p15 > G.pre_hi)) | ((d < 0) & (G.p15 < G.pre_lo))
            gross = d * ((1 + G.r240) / (1 + G.r15) - 1)
            for yr, idx in G.groupby(G.t.dt.year).groups.items():
                row(P, f"{yr}", gross.loc[idx][brk.loc[idx]])
            P("")


def section_B(P, px_by_sym, ge):
    P("# B. «Стрэддл» отложенными ордерами, выставленными до релиза\n")
    P("За минуту до релиза ставим buy-stop на max(30 мин до релиза) + буфер и sell-stop на min − буфер. "
      "Сработавший ордер открывает позицию, второй отменяется; стоп-лосс на противоположной границе; "
      "выход по времени. Не требует скорости: ордера стоят в книге заранее. Издержки 0.10%.\n")
    for sym, px in px_by_sym.items():
        P(f"## {sym}\n")
        P("| Выборка | буфер | выход | N сработало | доля срабатываний | gross % | медиана % | t | hit | net @0.10% | макс. убыток % |")
        P("|---|---|---|---|---|---|---|---|---|---|---|")
        for grp, mask in [("CPI", ge.is_cpi), ("FOMC", ge.is_fomc), ("NFP", ge.is_nfp), ("все релизы", ge.is_cpi | ~ge.is_cpi)]:
            G = ge[mask]
            for buf in [0.0005, 0.0015]:
                for kx in [60, 240, 1440]:
                    res = []
                    for _, e in G.iterrows():
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
                                trig = None; break      # обе границы в одной свече: неопределённость, пропускаем
                            if px.h[b] >= up:
                                trig = (+1, b, up); break
                            if px.l[b] <= dn:
                                trig = (-1, b, dn); break
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
                        res.append(r)
                    res = np.array(res)
                    if len(res) < 5:
                        continue
                    t, n = tstat(res)
                    P(f"| {grp} | {buf*100:.2f}% | {kx}м | {n} | {n/len(G)*100:.0f}% | {res.mean()*100:+.3f} | {np.median(res)*100:+.3f} | "
                      f"{t:+.2f} | {(res>0).mean()*100:.0f}% | {(res.mean()-COST)*100:+.3f} | {res.min()*100:.2f} |")
        P("")


def section_C(P, feats_by_sym):
    P("# C. Дневные горизонты вокруг запланированных событий\n")
    for sym, F in feats_by_sym.items():
        P(f"## {sym}\n")
        P("| Тест | N | средн. % | медиана % | t | hit |")
        P("|---|---|---|---|---|---|")
        for grp, mask in [("FOMC", F.is_fomc), ("CPI", F.is_cpi), ("NFP", F.is_nfp)]:
            G = F[mask]
            t, n = tstat(G.r_pre24)
            P(f"| {grp}: доходность за 24ч ДО релиза (pre-announcement drift) | {n} | {G.r_pre24.mean()*100:+.3f} | {G.r_pre24.median()*100:+.3f} | {t:+.2f} | {(G.r_pre24>0).mean()*100:.0f}% |")
            t, n = tstat(G.r1440)
            P(f"| {grp}: доходность за 24ч ПОСЛЕ релиза (long) | {n} | {G.r1440.mean()*100:+.3f} | {G.r1440.median()*100:+.3f} | {t:+.2f} | {(G.r1440>0).mean()*100:.0f}% |")
            # продолжение после первого часа на сутки
            d = np.sign(G.r60); g = d * ((1 + G.r1440) / (1 + G.r60) - 1)
            t, n = tstat(g)
            P(f"| {grp}: продолжение движения первого часа до +24ч (вход t+60) | {n} | {g.mean()*100:+.3f} | {g.median()*100:+.3f} | {t:+.2f} | {(g>0).mean()*100:.0f}% |")
            big = G.r60.abs() > 0.01
            t, n = tstat(g[big])
            P(f"| {grp}: то же, только если \\|r60\\| > 1% | {n} | {g[big].mean()*100:+.3f} | {g[big].median()*100:+.3f} | {t:+.2f} | {(g[big]>0).mean()*100:.0f}% |")
            # продолжение после 4 часов на сутки
            d4 = np.sign(G.r240); g4 = d4 * ((1 + G.r1440) / (1 + G.r240) - 1)
            t, n = tstat(g4)
            P(f"| {grp}: продолжение движения первых 4ч до +24ч (вход t+240) | {n} | {g4.mean()*100:+.3f} | {g4.median()*100:+.3f} | {t:+.2f} | {(g4>0).mean()*100:.0f}% |")
            # по сюрпризу на сутки
            if G.zc.notna().any():
                s = G[G.zc.abs() >= 0.5]; gs = np.sign(s.zc) * ((1 + s.r1440) / (1 + s.r60) - 1)
                t, n = tstat(gs)
                P(f"| {grp}: по знаку сюрприза, вход t+60, выход t+24ч | {n} | {gs.mean()*100:+.3f} | {gs.median()*100:+.3f} | {t:+.2f} | {(gs>0).mean()*100:.0f}% |")
        P("")


# ----------------------------------------------------------------------------------------------------
def load_news(path):
    df = pd.read_csv(path, na_values=["NULL"], low_memory=False)
    df["t"] = pd.to_datetime(df["newsDatetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["t"])
    df = df[df.t >= "2021-01-01"]
    df["cur"] = df["currencies"].fillna("")
    df["votes"] = df.positive + df.negative + df.important
    df["sent"] = df.positive - df.negative
    df["is_twitter"] = df.sourceDomain.eq("twitter.com")
    return df


def section_D(P, px_by_sym, news):
    P("# D. Внеплановые крипто-новости (CryptoPanic, 2021-2025)\n")
    P("Источник: 248k новостей CryptoPanic с секундными метками и голосами пользователей (positive / negative / important). "
      "Важно: голоса накапливаются ПОСЛЕ публикации, поэтому фильтр по голосам содержит заглядывание в будущее; "
      "результаты по направлению — верхняя граница того, что мог бы получить бот с идеальным классификатором. "
      "Метка времени CryptoPanic отстаёт от первоисточника, поэтому горизонты короче 15 минут не рассматриваем.\n")
    for sym, tag in [("BTCUSDT", "BTC"), ("ETHUSDT", "ETH"), ("SOLUSDT", "SOL")]:
        px = px_by_sym[sym]
        n = news[news.cur.str.split(",").apply(lambda xs: tag in xs)].copy()
        n = n.sort_values("t")
        P(f"## {sym}: новостей с тегом {tag}: {len(n)}\n")

        def rets(sub):
            rows = []
            for _, e in sub.iterrows():
                t0 = e["t"].floor("min")
                r = {"t": t0, "sent": e["sent"], "votes": e["votes"], "important": e["important"], "tw": e["is_twitter"]}
                for k in [15, 60, 240, 1440]:
                    r[f"r{k}"] = px.ret(t0, 0, k)
                r["r15_60"] = px.ret(t0, 15, 60); r["r15_240"] = px.ret(t0, 15, 240); r["r15_1440"] = px.ret(t0, 15, 1440)
                r["r60_1440"] = px.ret(t0, 60, 1440)
                r["ctl_r60"] = px.ret(t0 - pd.Timedelta(days=1), 0, 60); r["ctl_r240"] = px.ret(t0 - pd.Timedelta(days=1), 0, 240)
                rows.append(r)
            return pd.DataFrame(rows)

        # 1) Двигают ли цену «важные» новости: important >= 5, дедуп по 60 мин
        imp = n[n.important >= 5].copy()
        imp = imp[imp.t.diff().dt.total_seconds().fillna(1e9) > 3600]
        R = rets(imp)
        P(f"**Важные новости (important >= 5, не чаще 1 в час): N={len(R)}**\n")
        P("| Горизонт | медиана \\|r\\| событие % | медиана \\|r\\| контроль % | Mann-Whitney p |")
        P("|---|---|---|---|")
        for k in [60, 240]:
            mw = stats.mannwhitneyu(R[f"r{k}"].abs().dropna(), R[f"ctl_r{k}"].abs().dropna(), alternative="greater")
            P(f"| {k} мин | {R[f'r{k}'].abs().median()*100:.3f} | {R[f'ctl_r{k}'].abs().median()*100:.3f} | {mw.pvalue:.3f} |")
        P("")

        # 2) Направление по голосам: sent >= +5 (бычьи) vs <= -5 (медвежьи), дедуп 60 мин
        P("**Направление по сентименту голосов (positive − negative), дедуп 1 час, вход в t+15 (без гонки):**\n")
        P("| Группа | N | r(0→15) % | r(15→60) % | t | r(15→240) % | t | r(15→1440) % | t | hit 15→240 |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        for name, sub in [("бычьи (sent>=+5)", n[n.sent >= 5]), ("медвежьи (sent<=-5)", n[n.sent <= -5]),
                          ("бычьи сильные (sent>=+15)", n[n.sent >= 15]), ("медвежьи сильные (sent<=-10)", n[n.sent <= -10]),
                          ("бычьи, не twitter", n[(n.sent >= 5) & ~n.is_twitter]), ("медвежьи, не twitter", n[(n.sent <= -5) & ~n.is_twitter])]:
            sub = sub.sort_values("t"); sub = sub[sub.t.diff().dt.total_seconds().fillna(1e9) > 3600]
            if len(sub) < 10:
                continue
            S = rets(sub)
            sgn = 1 if "бычьи" in name else -1
            t1, _ = tstat(sgn * S.r15_60); t2, _ = tstat(sgn * S.r15_240); t3, _ = tstat(sgn * S.r15_1440)
            P(f"| {name} | {len(S)} | {S.r15.mean()*100:+.3f} | {S.r15_60.mean()*100:+.3f} | {t1:+.2f} | {S.r15_240.mean()*100:+.3f} | {t2:+.2f} | "
              f"{S.r15_1440.mean()*100:+.3f} | {t3:+.2f} | {(sgn*S.r15_240>0).mean()*100:.0f}% |")
        P("")

        # 3) Подтверждение: сентимент + цена уже пошла в ту же сторону за 15 мин
        P("**Сентимент + техническое подтверждение (цена за первые 15 мин пошла в сторону сентимента на > 0.2%):**\n")
        P("| Группа | N | r(15→240) % | t | hit | r(15→1440) % | t | net @0.10% (15→240) |")
        P("|---|---|---|---|---|---|---|---|")
        for name, sub, sgn in [("бычьи (sent>=+5)", n[n.sent >= 5], 1), ("медвежьи (sent<=-5)", n[n.sent <= -5], -1)]:
            sub = sub.sort_values("t"); sub = sub[sub.t.diff().dt.total_seconds().fillna(1e9) > 3600]
            S = rets(sub)
            conf = sgn * S.r15 > 0.002
            anti = sgn * S.r15 < -0.002
            for lab, m in [("подтверждено", conf), ("цена пошла ПРОТИВ новости", anti)]:
                g = sgn * S.r15_240[m]; g2 = sgn * S.r15_1440[m]
                t, cnt = tstat(g); t2, _ = tstat(g2)
                if cnt >= 10:
                    P(f"| {name}, {lab} | {cnt} | {g.mean()*100:+.3f} | {t:+.2f} | {(g>0).mean()*100:.0f}% | {g2.mean()*100:+.3f} | {t2:+.2f} | {(g.mean()-COST)*100:+.3f} |")
        P("")

        # 4) Интенсивность потока новостей как сигнал волатильности
        P("**Всплеск потока новостей (число новостей по монете за час >= 99-й перцентиль) как сигнал волатильности:**\n")
        cnt = n.set_index("t").assign(one=1)["one"].resample("1h").sum()
        thr = cnt.quantile(0.99)
        bursts = cnt[cnt >= thr]
        bursts = bursts[bursts.index.to_series().diff().dt.total_seconds().fillna(1e9) > 6 * 3600]
        rows = []
        for t0 in bursts.index:
            t1 = t0 + pd.Timedelta(hours=1)   # сигнал известен в конце часа
            rows.append({"r240": px.ret(t1, 0, 240), "r1440": px.ret(t1, 0, 1440),
                         "ctl240": px.ret(t1 - pd.Timedelta(days=1), 0, 240), "ctl1440": px.ret(t1 - pd.Timedelta(days=1), 0, 1440),
                         "pre": px.ret(t0, -240, 60)})
        B = pd.DataFrame(rows)
        if len(B) >= 10:
            P(f"- N всплесков: {len(B)} (порог >= {thr:.0f} новостей/час)")
            for k in [240, 1440]:
                mw = stats.mannwhitneyu(B[f"r{k}"].abs().dropna(), B[f"ctl{k}"].abs().dropna(), alternative="greater")
                P(f"- |r{k}| после всплеска: медиана {B[f'r{k}'].abs().median()*100:.2f}% vs контроль {B[f'ctl{k}'].abs().median()*100:.2f}% (p={mw.pvalue:.3f})")
            d = np.sign(B.pre); g = d * B.r1440
            t, cnt_ = tstat(g)
            P(f"- продолжение направления движения во время всплеска на следующие 24ч: средн. {g.mean()*100:+.3f}%, t={t:+.2f}, hit {(g>0).mean()*100:.0f}%")
            g2 = -d * B.r1440; t2, _ = tstat(g2)
            P(f"- разворот (fade) того же движения на 24ч: средн. {g2.mean()*100:+.3f}%, t={t2:+.2f}")
        P("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc", required=True); ap.add_argument("--eth", required=True); ap.add_argument("--sol", required=True)
    ap.add_argument("--events", required=True); ap.add_argument("--news", required=True)
    ap.add_argument("--out", default="research/results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lines = []; P = lines.append

    px_by_sym = {"BTCUSDT": Px(a.btc, "BTCUSDT"), "ETHUSDT": Px(a.eth, "ETHUSDT"), "SOLUSDT": Px(a.sol, "SOLUSDT")}
    for s, p in px_by_sym.items():
        print(s, pd.to_datetime(p.ts[0], unit="m"), pd.to_datetime(p.ts[-1], unit="m"), len(p.ts))
    events = load_events(a.events)
    ge = grouped_events(events)
    print("grouped events:", len(ge))
    feats = {s: features(p, ge) for s, p in px_by_sym.items()}
    for s, F in feats.items():
        F.to_csv(os.path.join(a.out, f"phase2_features_{s}.csv"), index=False)

    section_A(P, feats)
    section_B(P, px_by_sym, ge)
    section_C(P, feats)
    news = load_news(a.news)
    print("news since 2021:", len(news))
    section_D(P, px_by_sym, news)

    txt = "\n".join(lines)
    open(os.path.join(a.out, "results_phase2.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
