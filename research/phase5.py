"""
Фаза 5: позиционирование и «реакция относительно ожидаемого».

Позиционирование измеряется потоком агрессивных ордеров из минутных данных Binance:
  taker buy ratio = объём, исполненный покупателями по рынку / общий объём  (0.5 = баланс).
Это прямой показатель того, кто давит на цену, доступный за 2021-2026 без внешних данных.

  A. Поток до плановых событий (CPI, FOMC, NFP): предсказывает ли перекос за 24ч/7д направление и
     меняет ли он работу правила FOMC (вход в 14:30 по реакции на заявление).
  B. Поток в окне реакции как подтверждение: согласие цены и потока на заявлении FOMC.
  C. Поток в шоках без контекста (вселенная фазы 4): шок против накопленной позиции vs по ней.
  D. Реакция относительно ожидаемого: величина реакции на CPI и FOMC против своей истории и против сюрприза.

Запуск: python3 research/phase5.py --btc DIR --eth DIR --sol DIR --events events.json --out research/results
"""
import argparse, glob, os
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy import stats
from event_study import load_events, SIGN
from phase2 import grouped_events
from phase4 import fomc_times, cell, tstat, shock_universe, THR

NY = ZoneInfo("America/New_York")
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


class PxF:
    """Минутные бары с объёмом тейкер-покупок."""

    def __init__(self, data_dir, sym):
        files = sorted(glob.glob(os.path.join(data_dir, f"{sym}_1m_*.csv")))
        df = pd.concat([pd.read_csv(f, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_volume"]) for f in files])
        df = df.drop_duplicates("timestamp").sort_values("timestamp")
        self.sym = sym
        self.ts = df["timestamp"].to_numpy(np.int64) // 60000
        self.o = df["open"].to_numpy(float); self.h = df["high"].to_numpy(float); self.l = df["low"].to_numpy(float)
        self.c = df["close"].to_numpy(float); self.v = df["volume"].to_numpy(float); self.tb = df["taker_buy_volume"].to_numpy(float)
        self.cv = np.concatenate([[0.0], np.cumsum(self.v)]); self.ctb = np.concatenate([[0.0], np.cumsum(self.tb)])

    def idx(self, t):
        return int(np.searchsorted(self.ts, int(pd.Timestamp(t).value // 60_000_000_000)))

    def px(self, t):
        i = self.idx(t); return self.o[i] if i < len(self.o) else np.nan

    def ret(self, t0, k_from, k_to):
        a = self.px(t0 + pd.Timedelta(minutes=k_from)); b = self.px(t0 + pd.Timedelta(minutes=k_to)); return b / a - 1

    def flow(self, t0, k_from, k_to):
        """Taker buy ratio в окне [t0+k_from, t0+k_to)."""
        i, j = self.idx(t0 + pd.Timedelta(minutes=k_from)), self.idx(t0 + pd.Timedelta(minutes=k_to))
        if j <= i: return np.nan
        v = self.cv[j] - self.cv[i]; return (self.ctb[j] - self.ctb[i]) / v if v > 0 else np.nan

    def vol(self, t0, k_from, k_to):
        i, j = self.idx(t0 + pd.Timedelta(minutes=k_from)), self.idx(t0 + pd.Timedelta(minutes=k_to))
        return self.cv[j] - self.cv[i] if j > i else np.nan

    def rng(self, t0, k_from, k_to):
        i, j = self.idx(t0 + pd.Timedelta(minutes=k_from)), self.idx(t0 + pd.Timedelta(minutes=k_to))
        return (self.h[i:j].max(), self.l[i:j].min()) if j > i else (np.nan, np.nan)


def flow_z(px, t0, k_from, k_to, lookback_days=30):
    """Перекос потока в окне относительно распределения таких же окон за предыдущие lookback_days (z-score)."""
    f = px.flow(t0, k_from, k_to)
    w = k_to - k_from
    hist = [px.flow(t0 - pd.Timedelta(days=d), k_from, k_to) for d in range(1, lookback_days + 1)]
    hist = np.array([h for h in hist if not np.isnan(h)])
    if len(hist) < 10 or hist.std() == 0: return np.nan
    return (f - hist.mean()) / hist.std()


def section_A(P, px_by_sym, ge):
    P("# A. Поток ордеров ДО плановых событий как мера позиционирования\n")
    P("flow z = перекос taker-buy ratio за окно до события относительно таких же окон за предыдущие 30 дней. "
      "Положительный z = агрессивно покупали до события (толпа в лонге), отрицательный = продавали.\n")
    F = fomc_times()
    for s in SYMS:
        px = px_by_sym[s]
        P(f"## {s}\n")
        # FOMC
        R = F.copy()
        R["z24"] = [flow_z(px, t, -1440, 0) for t in R.t]
        R["z7d"] = [px.flow(t, -10080, 0) - 0.5 for t in R.t]
        R["r_stmt"] = [px.ret(t, 0, 30) for t in R.t]
        R["x1440"] = [px.ret(t, 30, 1440) for t in R.t]; R["x2880"] = [px.ret(t, 30, 2880) for t in R.t]
        R["r60"] = [px.ret(t, 0, 60) for t in R.t]; R["r1440"] = [px.ret(t, 0, 1440) for t in R.t]
        P("**FOMC**: предсказывает ли поток до заседания направление реакции и суток?\n")
        for lab, x in [("flow z за 24ч до", R.z24), ("taker-buy ratio − 0.5 за 7д до", R.z7d)]:
            s1 = stats.spearmanr(x, R.r_stmt, nan_policy="omit"); s2 = stats.spearmanr(x, R.r1440, nan_policy="omit")
            P(f"- {lab}: Spearman с реакцией на заявление {s1.statistic:+.2f} (p={s1.pvalue:.2f}), с доходностью за сутки {s2.statistic:+.2f} (p={s2.pvalue:.2f})")
        d = np.sign(R.r_stmt)
        P("\nПравило «вход 14:30 по реакции на заявление» с учётом позиционирования (средн. % / t / hit):\n")
        P("| Разрез | N | +24ч | +48ч |")
        P("|---|---|---|---|")
        for lab, m in [("все", R.r_stmt.notna()),
                       ("реакция ПРОТИВ позиционирования 24ч (z24·d < 0): толпу выбивает", d * R.z24 < 0),
                       ("реакция ПО позиционированию (z24·d > 0)", d * R.z24 > 0),
                       ("сильное позиционирование |z24| > 1, против", (R.z24.abs() > 1) & (d * R.z24 < 0)),
                       ("сильное позиционирование |z24| > 1, по", (R.z24.abs() > 1) & (d * R.z24 > 0))]:
            g = R[m]; dd = np.sign(g.r_stmt)
            P(f"| {lab} | {len(g)} | {cell(dd*g.x1440)} | {cell(dd*g.x2880)} |")
        P("")
        # CPI / NFP: pre-flow vs surprise-direction outcome
        for lab, mask in [("CPI", ge.is_cpi), ("NFP", ge.is_nfp)]:
            G = ge[mask].copy()
            G["z24"] = [flow_z(px, t, -1440, 0) for t in G.t]
            G["r60"] = [px.ret(t, 0, 60) for t in G.t]; G["r1440"] = [px.ret(t, 0, 1440) for t in G.t]
            G["x60_1440"] = [px.ret(t, 60, 1440) for t in G.t]
            s1 = stats.spearmanr(G.z24, G.r60, nan_policy="omit"); s2 = stats.spearmanr(G.z24, G.r1440, nan_policy="omit")
            P(f"**{lab}**: flow z за 24ч до → Spearman с r60 {s1.statistic:+.2f} (p={s1.pvalue:.2f}), с r1440 {s2.statistic:+.2f} (p={s2.pvalue:.2f}). ")
            # sign of surprise vs crowd
            sg = np.sign(G.zc); m_against = (sg * G.z24 < 0) & G.zc.notna() & (G.zc.abs() >= 0.5); m_with = (sg * G.z24 > 0) & G.zc.notna() & (G.zc.abs() >= 0.5)
            P(f"Сделка по знаку сюрприза (вход t+60, выход +24ч): сюрприз ПРОТИВ позиционирования {cell(sg[m_against]*G.x60_1440[m_against])}; "
              f"сюрприз ПО позиционированию {cell(sg[m_with]*G.x60_1440[m_with])}\n")
        P("")


def section_B(P, px_by_sym):
    P("# B. Поток в окне реакции на заявление FOMC как подтверждение\n")
    P("Направление по цене (знак r 14:00-14:30) и направление по потоку (taker-buy ratio в окне минус медиана за 30 дней). "
      "Гипотеза: когда цена и агрессивный поток согласны, продолжение сильнее.\n")
    F = fomc_times()
    for s in SYMS:
        px = px_by_sym[s]; R = F.copy()
        R["r_stmt"] = [px.ret(t, 0, 30) for t in R.t]
        R["f_stmt_z"] = [flow_z(px, t, 0, 30) for t in R.t]
        R["x1440"] = [px.ret(t, 30, 1440) for t in R.t]; R["x2880"] = [px.ret(t, 30, 2880) for t in R.t]
        R["volx"] = [px.vol(t, 0, 30) / (px.vol(t, -1440, 0) / 48) for t in R.t]
        d = np.sign(R.r_stmt); fd = np.sign(R.f_stmt_z)
        P(f"## {s}\n")
        P("| Разрез | N | +24ч (по цене) | +48ч (по цене) |")
        P("|---|---|---|---|")
        for lab, m in [("все", R.r_stmt.notna()), ("цена и поток согласны", d == fd), ("цена и поток расходятся", d != fd),
                       ("согласны и |flow z| > 1", (d == fd) & (R.f_stmt_z.abs() > 1)),
                       ("объём в окне > 5x нормы", R.volx > 5), ("объём в окне < 3x нормы", R.volx < 3)]:
            g = R[m]; dd = np.sign(g.r_stmt)
            P(f"| {lab} | {len(g)} | {cell(dd*g.x1440)} | {cell(dd*g.x2880)} |")
        sp = stats.spearmanr(R.f_stmt_z, R.x1440, nan_policy="omit")
        P(f"\nSpearman между flow z в окне заявления и доходностью 14:30 → +24ч: {sp.statistic:+.2f} (p={sp.pvalue:.2f}); "
          f"для сравнения, по цене r_stmt: {stats.spearmanr(R.r_stmt, R.x1440, nan_policy='omit').statistic:+.2f}\n")


def section_C(P, px_by_sym):
    P("# C. Поток в ценовых шоках без контекста (вселенная фазы 4)\n")
    P("Шок = |r60| > порога. pre-flow z = перекос потока за 24ч до шока; shock-flow = taker-buy ratio в шоковом часе − 0.5. "
      "Вход через 60 мин после шока.\n")
    for s in SYMS:
        px = px_by_sym[s]
        U = shock_universe(px, THR[s])
        U["pre_z"] = [flow_z(px, t, -1500, -60) for t in U.t]
        U["shock_flow"] = [px.flow(t, -60, 0) - 0.5 for t in U.t]
        U["post_flow"] = [px.flow(t, 0, 60) - 0.5 for t in U.t]
        d = U.d
        P(f"## {s} (N={len(U)})\n")
        P("Продолжение d × доходность от входа t+60 (средн. % / t / hit):\n")
        P("| Разрез | N | +4ч | +1д | +3д |")
        P("|---|---|---|---|---|")
        for lab, m in [("все", U.d.notna()),
                       ("шок ПРОТИВ позиционирования 24ч (d·pre_z < −1): выбивание толпы", d * U.pre_z < -1),
                       ("шок ПО позиционированию (d·pre_z > 1)", d * U.pre_z > 1),
                       ("шок с агрессивным потоком в ту же сторону (d·shock_flow > +0.05)", d * U.shock_flow > 0.05),
                       ("шок без агрессивного потока в ту сторону (d·shock_flow < 0)", d * U.shock_flow < 0),
                       ("поток в час после шока продолжает (d·post_flow > +0.03)", d * U.post_flow > 0.03),
                       ("поток в час после шока разворачивается (d·post_flow < −0.03)", d * U.post_flow < -0.03),
                       ("шок вниз, выбивание лонгов (pre_z > 1)", (d < 0) & (U.pre_z > 1)),
                       ("шок вверх, выбивание шортов (pre_z < −1)", (d > 0) & (U.pre_z < -1))]:
            g = U[m]
            P(f"| {lab} | {len(g)} | {cell(g.d*g.e_4h)} | {cell(g.d*g.e_1d)} | {cell(g.d*g.e_3d)} |")
        P("")


def section_D(P, px_by_sym, ge):
    P("# D. Реакция относительно ожидаемого\n")
    P("Ожидаемая реакция = медиана |r30| на предыдущих 12 релизах той же серии. ratio = |r30| текущий / ожидаемый. "
      "Для CPI дополнительно: соответствие величины реакции величине сюрприза.\n")
    F = fomc_times()
    for s in SYMS:
        px = px_by_sym[s]
        P(f"## {s}\n")
        # FOMC
        R = F.copy().reset_index(drop=True)
        R["r30"] = [px.ret(t, 0, 30) for t in R.t]; R["x1440"] = [px.ret(t, 30, 1440) for t in R.t]; R["x2880"] = [px.ret(t, 30, 2880) for t in R.t]
        R["x240"] = [px.ret(t, 30, 240) for t in R.t]
        exp_ = R.r30.abs().rolling(12, min_periods=6).median().shift(1)
        R["ratio"] = R.r30.abs() / exp_
        d = np.sign(R.r30)
        P("**FOMC** (вход 14:30 по реакции на заявление):\n")
        P("| Реакция относительно ожидаемой | N | +4ч | +24ч | +48ч |")
        P("|---|---|---|---|---|")
        for lab, m in [("слабая (ratio < 0.5)", R.ratio < 0.5), ("обычная (0.5–2)", (R.ratio >= 0.5) & (R.ratio <= 2)), ("сильная (ratio > 2)", R.ratio > 2)]:
            g = R[m]; dd = np.sign(g.r30)
            P(f"| {lab} | {len(g)} | {cell(dd*g.x240)} | {cell(dd*g.x1440)} | {cell(dd*g.x2880)} |")
        P("")
        # CPI
        G = ge[ge.is_cpi].copy().sort_values("t").reset_index(drop=True)
        G["r30"] = [px.ret(t, 0, 30) for t in G.t]; G["x1440"] = [px.ret(t, 30, 1440) for t in G.t]; G["x240"] = [px.ret(t, 30, 240) for t in G.t]
        exp_ = G.r30.abs().rolling(12, min_periods=6).median().shift(1); G["ratio"] = G.r30.abs() / exp_
        G["agree"] = np.sign(G.r30) == np.sign(G.zc)
        P("**CPI** (вход t+30 по направлению первых 30 минут):\n")
        P("| Разрез | N | +4ч | +24ч |")
        P("|---|---|---|---|")
        d = np.sign(G.r30)
        for lab, m in [("слабая реакция (ratio < 0.5)", G.ratio < 0.5), ("обычная", (G.ratio >= 0.5) & (G.ratio <= 2)), ("сильная (ratio > 2)", G.ratio > 2),
                       ("сильный сюрприз (|z|>1), слабая реакция (ratio<1)", (G.zc.abs() > 1) & (G.ratio < 1)),
                       ("сильный сюрприз (|z|>1), сильная реакция (ratio>1)", (G.zc.abs() > 1) & (G.ratio > 1)),
                       ("слабый сюрприз (|z|<0.5), сильная реакция (ratio>1.5): «реакция без причины»", (G.zc.abs() < 0.5) & (G.ratio > 1.5)),
                       ("реакция согласна с сюрпризом", G.agree & G.zc.notna()), ("реакция против сюрприза", ~G.agree & G.zc.notna())]:
            g = G[m]; dd = np.sign(g.r30)
            P(f"| {lab} | {len(g)} | {cell(dd*g.x240)} | {cell(dd*g.x1440)} |")
        P("")
        P("Сделка ПРОТИВ реакции (fade) при «реакции без причины» (|z|<0.5, ratio>1.5), вход t+30: "
          f"+4ч {cell(-d[(G.zc.abs() < 0.5) & (G.ratio > 1.5)] * G.x240[(G.zc.abs() < 0.5) & (G.ratio > 1.5)])}, "
          f"+24ч {cell(-d[(G.zc.abs() < 0.5) & (G.ratio > 1.5)] * G.x1440[(G.zc.abs() < 0.5) & (G.ratio > 1.5)])}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc", required=True); ap.add_argument("--eth", required=True); ap.add_argument("--sol", required=True)
    ap.add_argument("--events", required=True); ap.add_argument("--out", default="research/results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    px = {"BTCUSDT": PxF(a.btc, "BTCUSDT"), "ETHUSDT": PxF(a.eth, "ETHUSDT"), "SOLUSDT": PxF(a.sol, "SOLUSDT")}
    ge = grouped_events(load_events(a.events))
    L = []; P = L.append
    section_A(P, px, ge); section_B(P, px); section_C(P, px); section_D(P, px, ge)
    txt = "\n".join(L); open(os.path.join(a.out, "results_phase5.md"), "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
