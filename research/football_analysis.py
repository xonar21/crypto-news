"""
Футбол как тест идеи «посчитать вероятность из параметров и ставить на наивысшую, на длинной дистанции проигрыши — статистика».

Данные: Английская Премьер-лига 1993-2026, 12 704 матча, коэффициенты 24 букмекеров (AnishKhetani/premier-league-data,
источник football-data.co.uk). Bet365 с 2002, Pinnacle с 2012, лучшие коэффициенты рынка с 2005, закрытия с 2019.

  A. Маржа букмекера и что происходит с наивными стратегиями (всегда фаворит / аутсайдер / ничья / хозяева) за 20+ сезонов.
  B. Модели вероятностей из параметров команд (Elo с учётом разницы голов; Пуассон атака/защита с затуханием; логит на признаках),
     обучение только на прошлом, сравнение точности с букмекером (Brier, RPS, log loss) по сезонам.
  C. Добавляет ли модель информацию к коэффициентам (стэкинг: логит на [модель, букмекер]).
  D. Ставки по модели: «наивысшая вероятность» и value-ставки (p_model × коэффициент > 1 + порог) по лучшим коэффициентам рынка,
     доходность по сезонам, число ставок, закрывающая линия (CLV) как ранний индикатор.
  E. Математика длинной дистанции: симуляция банкролла при измеренной марже и дисперсии; сколько ставок нужно, чтобы отличить
     преимущество от удачи.
  F. Polymarket спорт (46k рынков): доходность покупки фаворитов/аутсайдеров по ценовым диапазонам после спреда.

Запуск: python3 research/football_analysis.py --odds results_with_odds.csv --pm observations.parquet --out research/results
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

OUT = "research/results"


def implied(h, d, a):
    """Де-вигованные вероятности (пропорциональная нормализация) и маржа."""
    inv = 1 / h + 1 / d + 1 / a
    return (1 / h) / inv, (1 / d) / inv, (1 / a) / inv, inv - 1


def brier3(P, y):
    """P: n×3 вероятности [H,D,A]; y: индексы 0/1/2."""
    Y = np.eye(3)[y]
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def rps(P, y):
    Y = np.eye(3)[y]
    cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
    return float(np.mean(np.sum((cp - cy)[:, :2] ** 2, axis=1) / 2))


def logloss(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-6, 1))))


# ---------------------------------------------------------------- модели
def elo_probs(df, k=20.0, home_adv=60.0):
    """Elo с множителем за разницу голов; вероятности 1X2 через упорядоченный логит, калиброванный на прошлом (walk-forward по сезонам)."""
    ratings = {}
    diffs = np.zeros(len(df))
    for i, r in enumerate(df.itertuples(index=False)):
        rh, ra = ratings.get(r.home_team, 1500.0), ratings.get(r.away_team, 1500.0)
        diffs[i] = rh + home_adv - ra
        eh = 1 / (1 + 10 ** (-(rh + home_adv - ra) / 400))
        sh = 1.0 if r.fthg > r.ftag else 0.5 if r.fthg == r.ftag else 0.0
        gd = abs(r.fthg - r.ftag); mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = k * mult * (sh - eh)
        ratings[r.home_team] = rh + delta; ratings[r.away_team] = ra - delta
    return diffs


def ordered_logit_fit(x, y):
    """y in {0:H,1:D,2:A}; ordered logit на x (чем больше x, тем вероятнее H). Возвращает (beta, c1, c2)."""
    def nll(theta):
        b, c1, c2 = theta[0], theta[1], theta[1] + np.exp(theta[2])
        z = b * x
        pA = 1 / (1 + np.exp(-(c1 - z)))          # P(y>=... ) ordering: A < D < H along z
        pAD = 1 / (1 + np.exp(-(c2 - z)))
        p = np.where(y == 2, pA, np.where(y == 1, pAD - pA, 1 - pAD))
        return -np.sum(np.log(np.clip(p, 1e-9, 1)))
    res = minimize(nll, x0=[0.005, -0.5, 0.0], method="Nelder-Mead", options={"maxiter": 4000})
    b, c1, c2 = res.x[0], res.x[1], res.x[1] + np.exp(res.x[2])
    return b, c1, c2


def ordered_logit_pred(x, b, c1, c2):
    z = b * x
    pA = 1 / (1 + np.exp(-(c1 - z))); pAD = 1 / (1 + np.exp(-(c2 - z)))
    return np.column_stack([1 - pAD, pAD - pA, pA])


def poisson_probs(df, halflife_days=365.0, max_goals=8):
    """Пуассон атака/защита с экспоненциальным затуханием, переоценка перед каждым туром (батчами по датам), walk-forward."""
    teams = sorted(set(df.home_team) | set(df.away_team)); idx = {t: i for i, t in enumerate(teams)}; T = len(teams)
    dates = df.date.to_numpy(); hi = df.home_team.map(idx).to_numpy(); ai = df.away_team.map(idx).to_numpy()
    hg = df.fthg.to_numpy(float); ag = df.ftag.to_numpy(float)
    P = np.full((len(df), 3), np.nan)
    params = np.zeros(2 * T + 1); params[-1] = 0.25
    uniq_dates = np.unique(dates)
    # переоценка раз в ~14 дней для скорости
    refit_points = uniq_dates[::7]
    last_fit = None
    for ref in refit_points:
        mask = dates < ref
        if mask.sum() < 600:
            continue
        w = 0.5 ** ((ref - dates[mask]).astype("timedelta64[D]").astype(float) / halflife_days)
        H, A, HG, AG = hi[mask], ai[mask], hg[mask], ag[mask]
        def nll(p):
            att, dfn, hadv = p[:T], p[T:2 * T], p[-1]
            lh = np.exp(att[H] + dfn[A] + hadv); la = np.exp(att[A] + dfn[H])
            return -np.sum(w * (HG * np.log(lh) - lh + AG * np.log(la) - la)) + 0.1 * (np.sum(att ** 2) + np.sum(dfn ** 2))
        def grad(p):
            att, dfn, hadv = p[:T], p[T:2 * T], p[-1]
            lh = np.exp(att[H] + dfn[A] + hadv); la = np.exp(att[A] + dfn[H])
            g = np.zeros_like(p)
            np.add.at(g, H, -w * (HG - lh)); np.add.at(g, A, -w * (AG - la))
            np.add.at(g, T + A, -w * (HG - lh)); np.add.at(g, T + H, -w * (AG - la))
            g[-1] = -np.sum(w * (HG - lh))
            g[:T] += 0.2 * att; g[T:2 * T] += 0.2 * dfn
            return g
        res = minimize(nll, params, jac=grad, method="L-BFGS-B", options={"maxiter": 200})
        params = res.x; last_fit = ref
        # предсказать матчи до следующей точки переоценки
        nxt = refit_points[refit_points > ref]
        upto = nxt[0] if len(nxt) else np.datetime64("2100-01-01")
        sel = np.where((dates >= ref) & (dates < upto))[0]
        att, dfn, hadv = params[:T], params[T:2 * T], params[-1]
        g = np.arange(max_goals + 1)
        for i in sel:
            lh = np.exp(att[hi[i]] + dfn[ai[i]] + hadv); la = np.exp(att[ai[i]] + dfn[hi[i]])
            ph = stats.poisson.pmf(g, lh); pa = stats.poisson.pmf(g, la)
            M = np.outer(ph, pa)
            P[i] = [np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()]
    return P


def multinomial_logit_fit_predict(Xtr, ytr, Xte):
    """Мультиномиальная логистическая регрессия (softmax) с L2, через scipy."""
    n, k = Xtr.shape; Xtr1 = np.column_stack([np.ones(n), Xtr]); Xte1 = np.column_stack([np.ones(len(Xte)), Xte])
    Y = np.eye(3)[ytr]
    def nll(wflat):
        W = wflat.reshape(k + 1, 3); Z = Xtr1 @ W; Z -= Z.max(1, keepdims=True)
        Pm = np.exp(Z); Pm /= Pm.sum(1, keepdims=True)
        return -np.sum(Y * np.log(np.clip(Pm, 1e-9, 1))) + 0.01 * np.sum(W[1:] ** 2)
    def grad(wflat):
        W = wflat.reshape(k + 1, 3); Z = Xtr1 @ W; Z -= Z.max(1, keepdims=True)
        Pm = np.exp(Z); Pm /= Pm.sum(1, keepdims=True)
        G = Xtr1.T @ (Pm - Y); G[1:] += 0.02 * W[1:]
        return G.ravel()
    res = minimize(nll, np.zeros((k + 1) * 3), jac=grad, method="L-BFGS-B")
    W = res.x.reshape(k + 1, 3); Z = Xte1 @ W; Z -= Z.max(1, keepdims=True); Pm = np.exp(Z)
    return Pm / Pm.sum(1, keepdims=True)


# ---------------------------------------------------------------- главный анализ
def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--odds", required=True); ap.add_argument("--pm", required=False); ap.add_argument("--out", default="research/results")
    a = ap.parse_args(); OUT = a.out; os.makedirs(OUT, exist_ok=True)
    L = []; P = L.append

    odds_df = pd.read_csv(a.odds, parse_dates=["date"])
    res_df = pd.read_csv(os.path.join(os.path.dirname(a.odds), "results.csv"), parse_dates=["date"])[["match_id", "season", "fthg", "ftag", "ftr"]]
    d = res_df.merge(odds_df.drop(columns=[c for c in ["season"] if c in odds_df.columns]), on="match_id", how="left")
    d = d.sort_values(["date", "match_id"]).reset_index(drop=True)
    d["y"] = d.ftr.map({"H": 0, "D": 1, "A": 2})
    # лучшие коэффициенты рынка: betbrain_max (2005-2019) + market_max (2019-)
    for o in ["home", "draw", "away"]:
        d[f"best_{o}"] = d[f"market_max_1x2_{o}"].fillna(d[f"betbrain_max_1x2_{o}"])
        d[f"avg_{o}"] = d[f"market_avg_1x2_{o}"].fillna(d[f"betbrain_avg_1x2_{o}"])
    P("# Футбол: проверка идеи «считать вероятность из параметров и ставить на наивысшую»\n")
    P(f"Данные: АПЛ, {len(d)} матчей {d.date.min().date()} — {d.date.max().date()}. Bet365 с 2002/03 ({int(d.bet365_1x2_home.notna().sum())} матчей), "
      f"Pinnacle с 2012/13 ({int(d.pinnacle_1x2_home.notna().sum())}), лучшие коэффициенты рынка с 2005/06 ({int(d.best_home.notna().sum())}), закрытия Pinnacle с 2019/20 ({int(d.pinnacle_1x2_home_close.notna().sum())}).\n")
    P(f"Базовые частоты: хозяева {np.mean(d.y==0)*100:.1f}%, ничья {np.mean(d.y==1)*100:.1f}%, гости {np.mean(d.y==2)*100:.1f}%.\n")

    # ---------------- A. маржа и наивные стратегии
    P("## A. Маржа букмекера и наивные стратегии\n")
    P("| Букмекер | матчей | средняя маржа | Brier де-вигованных вероятностей | RPS |")
    P("|---|---|---|---|---|")
    books = [("Bet365 (открытие)", "bet365_1x2"), ("Pinnacle (открытие)", "pinnacle_1x2"), ("Pinnacle (закрытие)", "pinnacle_1x2", "_close"),
             ("Bet365 (закрытие)", "bet365_1x2", "_close"), ("Лучшие коэф. рынка", "best"), ("Средние коэф. рынка", "avg")]
    for b in books:
        suf = b[2] if len(b) > 2 else ""
        if b[1] in ("best", "avg"):
            h, dr, aw = d[f"{b[1]}_home"], d[f"{b[1]}_draw"], d[f"{b[1]}_away"]
        else:
            h, dr, aw = d[f"{b[1]}_home{suf}"], d[f"{b[1]}_draw{suf}"], d[f"{b[1]}_away{suf}"]
        m = h.notna() & dr.notna() & aw.notna()
        ph, pd_, pa, marg = implied(h[m], dr[m], aw[m])
        Pm = np.column_stack([ph, pd_, pa]); y = d.y[m].to_numpy()
        P(f"| {b[0]} | {int(m.sum())} | {marg.mean()*100:.2f}% | {brier3(Pm, y):.4f} | {rps(Pm, y):.4f} |")
    P("")
    P("Наивные стратегии по коэффициентам Bet365 (ставка 1 ед. на каждый матч, 2002–2026) и по лучшим коэффициентам рынка (2005–2026):\n")
    P("| Стратегия | букмекер | ставок | доходность на ставку | итог за период | доля сезонов в плюсе |")
    P("|---|---|---|---|---|---|")
    for bname, pref in [("Bet365", "bet365_1x2_"), ("лучшие коэф.", "best_")]:
        h, dr, aw = d[pref + "home"], d[pref + "draw"], d[pref + "away"]
        m = h.notna() & dr.notna() & aw.notna(); g = d[m].copy(); H, D, A = h[m].to_numpy(), dr[m].to_numpy(), aw[m].to_numpy(); y = g.y.to_numpy()
        odds = np.column_stack([H, D, A]); fav = odds.argmin(1); dog = odds.argmax(1)
        def ret(sel):
            win = (sel == y); return win * odds[np.arange(len(y)), sel] - 1
        for lab, sel in [("всегда хозяева", np.zeros(len(y), int)), ("всегда ничья", np.ones(len(y), int)), ("всегда гости", np.full(len(y), 2)),
                         ("всегда фаворит (наивысшая вероятность)", fav), ("всегда аутсайдер", dog)]:
            r = ret(sel); g["_r"] = r
            by = g.groupby("season")["_r"].sum()
            P(f"| {lab} | {bname} | {len(r)} | {r.mean()*100:+.2f}% | {r.sum():+.0f} ед. | {(by>0).mean()*100:.0f}% |")
    P("")
    # favorite-longshot by implied prob bins (Bet365)
    P("Доходность по диапазонам вероятности (Bet365, все три исхода как отдельные ставки):\n")
    P("| Подразумеваемая вероятность | ставок | фактическая частота | средний коэф. | доходность на ставку |")
    P("|---|---|---|---|---|")
    h, dr, aw = d.bet365_1x2_home, d.bet365_1x2_draw, d.bet365_1x2_away; m = h.notna() & dr.notna() & aw.notna()
    odds = np.column_stack([h[m], dr[m], aw[m]]); y = d.y[m].to_numpy(); Y = np.eye(3)[y]
    pimp = (1 / odds) / (1 / odds).sum(1, keepdims=True)
    flat_p, flat_o, flat_y = pimp.ravel(), odds.ravel(), Y.ravel()
    for lo, hi_ in [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)]:
        mm = (flat_p >= lo) & (flat_p < hi_)
        if mm.sum() < 50: continue
        r = flat_y[mm] * flat_o[mm] - 1
        P(f"| {lo*100:.0f}–{min(hi_,1)*100:.0f}% | {int(mm.sum())} | {flat_y[mm].mean()*100:.1f}% | {flat_o[mm].mean():.2f} | {r.mean()*100:+.2f}% |")
    P("")

    # ---------------- B. модели
    P("## B. Модели вероятностей из параметров команд против букмекера (walk-forward, обучение только на прошлом)\n")
    d["elo_diff"] = elo_probs(d)
    P_pois = poisson_probs(d)
    # признаки для логита: elo_diff, форма (очки за последние 5 матчей), средние голы за/против за 10 матчей
    def rolling_team_feats(df):
        pts_h = np.where(df.y == 0, 3, np.where(df.y == 1, 1, 0)); pts_a = np.where(df.y == 2, 3, np.where(df.y == 1, 1, 0))
        hist = {}; fh = np.zeros((len(df), 3)); fa = np.zeros((len(df), 3))
        for i, r in enumerate(df.itertuples(index=False)):
            for team, arr in [(r.home_team, fh), (r.away_team, fa)]:
                hh = hist.get(team, [])
                if len(hh) >= 5:
                    last = hh[-10:]
                    arr[i] = [np.mean([x[0] for x in hh[-5:]]), np.mean([x[1] for x in last]), np.mean([x[2] for x in last])]
                else:
                    arr[i] = [np.nan, np.nan, np.nan]
            hist.setdefault(r.home_team, []).append((pts_h[i], r.fthg, r.ftag)); hist.setdefault(r.away_team, []).append((pts_a[i], r.ftag, r.fthg))
        return fh, fa
    fh, fa = rolling_team_feats(d)
    X = np.column_stack([d.elo_diff / 100, fh, fa])
    seasons = sorted(d.season.unique())
    P_elo = np.full((len(d), 3), np.nan); P_log = np.full((len(d), 3), np.nan)
    for s in seasons:
        te = (d.season == s).to_numpy(); tr = (d.date < d.date[te].min()).to_numpy() & ~np.isnan(X).any(1)
        if tr.sum() < 1500: continue
        b, c1, c2 = ordered_logit_fit(d.elo_diff[tr].to_numpy(), d.y[tr].to_numpy())
        P_elo[te] = ordered_logit_pred(d.elo_diff[te].to_numpy(), b, c1, c2)
        ok = te & ~np.isnan(X).any(1)
        P_log[ok] = multinomial_logit_fit_predict(X[tr], d.y[tr].to_numpy(), X[ok])
    d["P_elo"] = list(P_elo); d["P_pois"] = list(P_pois); d["P_log"] = list(P_log)
    # сравнение на матчах с Bet365 и всеми моделями
    hb, db, ab = d.bet365_1x2_home, d.bet365_1x2_draw, d.bet365_1x2_away
    mB = hb.notna() & db.notna() & ab.notna() & ~np.isnan(P_elo).any(1) & ~np.isnan(P_pois).any(1) & ~np.isnan(P_log).any(1)
    ph, pd_, pa, _ = implied(hb[mB], db[mB], ab[mB]); P_b365 = np.column_stack([ph, pd_, pa])
    y = d.y[mB].to_numpy()
    P(f"Общая выборка сравнения: {int(mB.sum())} матчей (2003/04–2025/26).\n")
    P("| Прогноз | Brier | RPS | log loss | точность (argmax) |")
    P("|---|---|---|---|---|")
    base = np.tile(np.bincount(y, minlength=3) / len(y), (len(y), 1))
    for lab, Pm in [("базовые частоты (константа)", base), ("Elo + упорядоченный логит", P_elo[mB]), ("Пуассон атака/защита (затухание 1 год)", P_pois[mB]),
                    ("Логит на признаках (Elo, форма, голы)", P_log[mB]), ("Bet365 де-вигованный", P_b365)]:
        P(f"| {lab} | {brier3(Pm, y):.4f} | {rps(Pm, y):.4f} | {logloss(Pm, y):.4f} | {np.mean(Pm.argmax(1)==y)*100:.1f}% |")
    P("")
    # по сезонам: RPS модели лучшей vs Bet365
    P("RPS по сезонам: лучшая модель (Пуассон) против Bet365 (меньше — лучше):\n")
    P("| Сезон | N | RPS Пуассон | RPS Elo | RPS Bet365 | модель лучше букмекера? |")
    P("|---|---|---|---|---|---|")
    wins = 0; tot = 0
    for s in seasons:
        ms = mB & (d.season == s)
        if ms.sum() < 100: continue
        ys = d.y[ms].to_numpy(); ph, pd_, pa, _ = implied(hb[ms], db[ms], ab[ms]); Pb = np.column_stack([ph, pd_, pa])
        r_p, r_e, r_b = rps(P_pois[ms], ys), rps(P_elo[ms], ys), rps(Pb, ys)
        wins += r_p < r_b; tot += 1
        P(f"| {s} | {int(ms.sum())} | {r_p:.4f} | {r_e:.4f} | {r_b:.4f} | {'да' if r_p < r_b else 'нет'} |")
    P(f"\nМодель точнее букмекера в {wins} из {tot} сезонов.\n")

    # ---------------- C. стэкинг
    P("## C. Добавляет ли модель информацию к коэффициентам (стэкинг, walk-forward)\n")
    P("Логит на [logit p_букмекера, logit p_модели] по каждому исходу; если вес модели близок к нулю и точность не растёт, информация модели уже в коэффициентах.\n")
    rows = []
    P_stack = np.full((len(d), 3), np.nan)
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    allB = np.full((len(d), 3), np.nan); allB[mB] = P_b365
    Z = np.column_stack([lg(allB), lg(P_pois)])
    for s in seasons:
        te = (d.season == s).to_numpy() & mB.to_numpy(); tr = (d.date < d.date[te].min()).to_numpy() & mB.to_numpy() if te.any() else None
        if tr is None or tr.sum() < 1500: continue
        P_stack[te] = multinomial_logit_fit_predict(Z[tr], d.y[tr].to_numpy(), Z[te])
    ms = mB & ~np.isnan(P_stack).any(1)
    ys = d.y[ms].to_numpy(); Pb = allB[ms]
    P(f"- Выборка: {int(ms.sum())} матчей. RPS Bet365 {rps(Pb, ys):.4f}; RPS стэкинга (букмекер + Пуассон) {rps(P_stack[ms], ys):.4f}; "
      f"RPS смеси 50/50 {rps(0.5*Pb+0.5*P_pois[ms], ys):.4f}; RPS смеси 80/20 {rps(0.8*Pb+0.2*P_pois[ms], ys):.4f}.")
    # Pinnacle closing as the strongest benchmark
    hc, dc, ac = d.pinnacle_1x2_home_close, d.pinnacle_1x2_draw_close, d.pinnacle_1x2_away_close
    mC = hc.notna() & dc.notna() & ac.notna() & ~np.isnan(P_pois).any(1)
    ph, pd_, pa, _ = implied(hc[mC], dc[mC], ac[mC]); Pc = np.column_stack([ph, pd_, pa]); yc = d.y[mC].to_numpy()
    P(f"- Против закрытия Pinnacle (2019–2026, {int(mC.sum())} матчей): RPS Pinnacle close {rps(Pc, yc):.4f}; Пуассон {rps(P_pois[mC], yc):.4f}; "
      f"Elo {rps(P_elo[mC], yc):.4f}; смесь 80/20 {rps(0.8*Pc+0.2*P_pois[mC], yc):.4f}.")
    # does opening->closing movement predict beyond closing? and does the model predict movement?
    ho, do_, ao = d.pinnacle_1x2_home, d.pinnacle_1x2_draw, d.pinnacle_1x2_away
    mO = mC & ho.notna()
    pho, _, _, _ = implied(ho[mO], do_[mO], ao[mO]); phc, _, _, _ = implied(hc[mO], dc[mO], ac[mO])
    move = phc.to_numpy() - pho.to_numpy(); model_edge = P_pois[mO.to_numpy(), 0] - pho.to_numpy()
    sp = stats.spearmanr(model_edge, move)
    P(f"- Предсказывает ли модель движение линии Pinnacle от открытия к закрытию (хозяева)? Spearman(p_модели − p_открытия, p_закрытия − p_открытия) = {sp.statistic:+.3f} (p={sp.pvalue:.3f}, N={int(mO.sum())}). "
      f"Медианный сдвиг линии |Δp| {np.median(np.abs(move))*100:.1f} п.п.")
    P("")

    # ---------------- D. ставки
    P("## D. Ставки по модели по лучшим коэффициентам рынка (2005–2026)\n")
    mD = d.best_home.notna() & ~np.isnan(P_pois).any(1)
    g = d[mD].copy(); odds = np.column_stack([g.best_home, g.best_draw, g.best_away]); Pm = P_pois[mD.to_numpy()]; y = g.y.to_numpy()
    pimp = (1 / odds) / (1 / odds).sum(1, keepdims=True)
    P("| Стратегия | ставок | доля матчей | доходность на ставку | итог, ед. | t-stat | доля сезонов в плюсе | худший сезон, ед. |")
    P("|---|---|---|---|---|---|---|---|")
    def report(lab, sel_mask, sel_out):
        idx = np.where(sel_mask)[0]
        if len(idx) < 30: P(f"| {lab} | {len(idx)} | — | — | — | — | — | — |"); return
        r = (y[idx] == sel_out[idx]) * odds[idx, sel_out[idx]] - 1
        gg = g.iloc[idx].copy(); gg["_r"] = r; by = gg.groupby("season")["_r"].sum()
        t = r.mean() / (r.std() / np.sqrt(len(r)))
        P(f"| {lab} | {len(r)} | {len(idx)/len(y)*100:.0f}% | {r.mean()*100:+.2f}% | {r.sum():+.0f} | {t:+.2f} | {(by>0).mean()*100:.0f}% | {by.min():+.0f} |")
    fav_model = Pm.argmax(1)
    report("наивысшая вероятность по модели, каждый матч", np.ones(len(y), bool), fav_model)
    report("наивысшая вероятность по модели, только если p > 0.6", Pm.max(1) > 0.6, fav_model)
    best_val = (Pm * odds).argmax(1); ev = (Pm * odds).max(1)
    for thr in [0.0, 0.05, 0.10, 0.20]:
        report(f"value: p_модели × коэф. > {1+thr:.2f} (лучший исход)", ev > 1 + thr, best_val)
    # value only on favorites / only on draws / only away
    report("value > 1.05 только хозяева", (Pm[:, 0] * odds[:, 0] > 1.05), np.zeros(len(y), int))
    report("value > 1.05 только ничьи", (Pm[:, 1] * odds[:, 1] > 1.05), np.ones(len(y), int))
    report("value > 1.05 только гости", (Pm[:, 2] * odds[:, 2] > 1.05), np.full(len(y), 2))
    # Elo as model
    Pe = P_elo[mD.to_numpy()]; ok = ~np.isnan(Pe).any(1)
    bv = np.where(ok, (np.nan_to_num(Pe) * odds).argmax(1), 0); eve = np.where(ok, (np.nan_to_num(Pe) * odds).max(1), 0)
    report("value > 1.05 по Elo-модели", eve > 1.05, bv)
    # consensus edge: best odds vs average odds (outlier odds strategy, Kaunitz-type)
    avg = np.column_stack([g.avg_home, g.avg_draw, g.avg_away]); okA = ~np.isnan(avg).any(1)
    pavg = (1 / np.where(okA[:, None], avg, 1)) ; pavg = pavg / pavg.sum(1, keepdims=True)
    ratio = np.where(okA[:, None], odds * pavg, 0)  # коэф. у лучшего букмекера × вероятность по среднему рынку
    sel = ratio.argmax(1); rv = ratio.max(1)
    for thr in [1.0, 1.03, 1.06]:
        report(f"выброс коэффициента: лучший коэф. × p_среднего рынка > {thr:.2f}", okA & (rv > thr), sel)
    P("")
    P("CLV (закрывающая линия) для value-ставок по модели в 2019–2026: средняя разница между подразумеваемой вероятностью закрытия Pinnacle и открытия для выбранного исхода.\n")
    mE = mD & mC & ho.notna()
    gE = d[mE]; Pm2 = P_pois[mE.to_numpy()]; oddsE = np.column_stack([gE.best_home, gE.best_draw, gE.best_away])
    bvE = (Pm2 * oddsE).argmax(1); evE = (Pm2 * oddsE).max(1)
    po = np.column_stack(implied(gE.pinnacle_1x2_home, gE.pinnacle_1x2_draw, gE.pinnacle_1x2_away)[:3]); pc = np.column_stack(implied(gE.pinnacle_1x2_home_close, gE.pinnacle_1x2_draw_close, gE.pinnacle_1x2_away_close)[:3])
    for thr in [1.0, 1.05, 1.10]:
        sel = evE > thr; i = np.arange(len(gE))[sel]
        clv = pc[i, bvE[i]] - po[i, bvE[i]]
        P(f"- порог {thr:.2f}: ставок {len(i)}, средний CLV {clv.mean()*100:+.2f} п.п., доля ставок с положительным CLV {(clv>0).mean()*100:.0f}%, t={clv.mean()/(clv.std()/np.sqrt(len(clv))):+.2f}")
    P("")

    # ---------------- E. математика длинной дистанции
    P("## E. Математика длинной дистанции\n")
    r_fav = None
    hB, dB, aB = d.bet365_1x2_home, d.bet365_1x2_draw, d.bet365_1x2_away; m = hB.notna() & dB.notna() & aB.notna()
    oddsB = np.column_stack([hB[m], dB[m], aB[m]]); yB = d.y[m].to_numpy(); favB = oddsB.argmin(1)
    r_fav = (yB == favB) * oddsB[np.arange(len(yB)), favB] - 1
    mu, sd = r_fav.mean(), r_fav.std()
    P(f"Ставка на фаворита Bet365: средняя доходность {mu*100:+.2f}% на ставку, стандартное отклонение {sd*100:.0f}%. ")
    rng = np.random.default_rng(1)
    for n in [100, 1000, 10000]:
        sims = rng.choice(r_fav, size=(5000, n)).sum(1)
        P(f"- после {n} ставок по 1 ед.: ожидание {mu*n:+.0f} ед., вероятность быть в плюсе {np.mean(sims>0)*100:.0f}%, 5-й перцентиль {np.percentile(sims,5):+.0f} ед.")
    P("")
    P("Сколько ставок нужно, чтобы отличить настоящее преимущество от удачи (t=2 при стандартном отклонении доходности одной ставки около 100%):\n")
    for edge in [0.01, 0.02, 0.03, 0.05, 0.10]:
        n = (2 * 1.0 / edge) ** 2
        P(f"- преимущество {edge*100:.0f}% на ставку: ≈ {n:,.0f} ставок (при 380 матчах АПЛ в сезон и ставке на 20% матчей это {n/76:.0f} сезонов)")
    P("")
    P("Симуляция банкролла при преимуществе +3% на ставку и ставке 2% банкролла (1000 ставок, 5000 траекторий):\n")
    base_r = rng.choice(r_fav, size=(5000, 1000)) - mu + 0.03
    eq = np.cumprod(1 + 0.02 * base_r, axis=1)
    dd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(1)
    P(f"- медианный итог ×{np.median(eq[:,-1]):.2f}, доля траекторий в минусе после 1000 ставок {np.mean(eq[:,-1]<1)*100:.0f}%, медианная максимальная просадка {np.median(dd)*100:.0f}%, 5-й перцентиль просадки {np.percentile(dd,5)*100:.0f}%\n")

    # ---------------- F. Polymarket спорт
    if a.pm and os.path.exists(a.pm):
        P("## F. Спорт на Polymarket (46k рынков): покупка по ценовым диапазонам после спреда\n")
        pm = pd.read_parquet(a.pm); sp_ = pm[(pm.platform == "Polymarket") & (pm.category == "Sports") & (pm.price > 0) & (pm.price < 1)]
        P("| Горизонт | диапазон цены Yes | N | частота Yes | доходность покупки Yes по цене + 1 п.п. | доходность покупки No |")
        P("|---|---|---|---|---|---|")
        for hzn in ["1d", "1h"]:
            g2 = sp_[sp_.horizon == hzn]
            for lo, hi_ in [(0, .1), (.1, .25), (.25, .4), (.4, .6), (.6, .75), (.75, .9), (.9, 1)]:
                mm = (g2.price >= lo) & (g2.price < hi_)
                if mm.sum() < 100: continue
                p = g2.price[mm].to_numpy(); yy = g2.outcome[mm].to_numpy()
                ry = (yy - (p + 0.01)) / (p + 0.01); rn = ((1 - yy) - (1 - p + 0.01)) / (1 - p + 0.01)
                P(f"| {hzn} | {lo*100:.0f}–{hi_*100:.0f}% | {int(mm.sum())} | {yy.mean()*100:.1f}% | {ry.mean()*100:+.1f}% | {rn.mean()*100:+.1f}% |")
        P("")

    txt = "\n".join(L); open(os.path.join(OUT, "results_football.md"), "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
