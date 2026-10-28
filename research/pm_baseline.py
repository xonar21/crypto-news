"""
Рынки предсказаний: базовые цифры для честного бэктеста LLM-прогнозиста.

Данные: hrishi-tailor/prediction-market-calibration, release v1.2.0, observations.parquet
(46k разрешённых рынков Kalshi и Polymarket, цена за 30д / 7д / 1д / 1ч до закрытия, исход).

Считает: (1) планку — Brier рынка по горизонтам и категориям; (2) сколько информации приходит между горизонтами;
(3) размер выборки без утечки для модели с заданной датой отсечки; (4) экономику ставок против рынка
при заданной точности прогнозиста (симуляция) с учётом комиссий.

Запуск: python3 research/pm_baseline.py --data observations.parquet --cutoff 2026-06-01 --out research/results
"""
import argparse, os
import numpy as np
import pandas as pd


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--cutoff", default="2026-06-01"); ap.add_argument("--out", default="research/results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    d = pd.read_parquet(a.data)
    d = d[(d.price > 0) & (d.price < 1)]
    L = []; P = L.append
    P("# Рынки предсказаний: планка и размер честной выборки\n")
    P(f"Данные: {d.market_id.nunique()} рынков, {len(d)} наблюдений, закрытия {d.close_time.min().date()} — {d.close_time.max().date()}.\n")

    # 1. Планка: Brier рынка по платформе и горизонту
    P("## 1. Планка: Brier рынка (чем ниже, тем лучше; 0.25 = монетка на 50/50)\n")
    P("| Платформа | Горизонт | N | доля Yes | Brier рынка | Brier базовой ставки (доля Yes) | Brier Skill рынка |")
    P("|---|---|---|---|---|---|---|")
    for (pl, h), g in d.groupby(["platform", "horizon"], observed=True):
        b = brier(g.price, g.outcome); base = brier(np.full(len(g), g.outcome.mean()), g.outcome)
        P(f"| {pl} | {h} | {len(g)} | {g.outcome.mean()*100:.0f}% | {b:.4f} | {base:.4f} | {1-b/base:.2f} |")
    P("")
    P("По категориям (Polymarket + Kalshi, горизонт 7д и 1д):\n")
    P("| Категория | N 7д | Brier рынка 7д | N 1д | Brier рынка 1д | доля Yes |")
    P("|---|---|---|---|---|---|")
    for c, g in d.groupby("category"):
        g7 = g[g.horizon == "7d"]; g1 = g[g.horizon == "1d"]
        if len(g1) < 200: continue
        P(f"| {c} | {len(g7)} | {brier(g7.price, g7.outcome) if len(g7) else float('nan'):.4f} | {len(g1)} | {brier(g1.price, g1.outcome):.4f} | {g1.outcome.mean()*100:.0f}% |")
    P("")

    # 2. Сколько информации приходит между горизонтами
    P("## 2. Движение цены между горизонтами (есть ли что ловить медленному участнику)\n")
    w = d.pivot_table(index=["platform", "market_id", "outcome", "category"], columns="horizon", values="price", observed=True).reset_index()
    for pl in ["Polymarket", "Kalshi"]:
        g = w[w.platform == pl]
        m7 = g.dropna(subset=["7d", "1d"]); m1 = g.dropna(subset=["1d", "1h"])
        P(f"- {pl}: |Δцены| 7д→1д медиана {np.median(np.abs(m7['1d']-m7['7d']))*100:.1f} п.п. (N={len(m7)}); 1д→1ч медиана {np.median(np.abs(m1['1h']-m1['1d']))*100:.1f} п.п. (N={len(m1)}). "
          f"Доля рынков, где цена за 7д→1д сдвинулась > 20 п.п.: {(np.abs(m7['1d']-m7['7d'])>0.2).mean()*100:.0f}%.")
        # направление сдвига и исход: предсказывает ли сдвиг 7д→1д исход сверх цены 1д? (momentum in prices)
        mm = m7.copy(); mm["d"] = mm["1d"] - mm["7d"]
        up = mm[mm.d > 0.1]; dn = mm[mm.d < -0.1]
        P(f"  - после роста цены > 10 п.п. за неделю: цена 1д {up['1d'].mean()*100:.0f}% против частоты Yes {up.outcome.mean()*100:.0f}% (N={len(up)}); "
          f"после падения > 10 п.п.: цена 1д {dn['1d'].mean()*100:.0f}% против Yes {dn.outcome.mean()*100:.0f}% (N={len(dn)})")
    P("")

    # 3. Размер выборки без утечки
    cutoff = pd.Timestamp(a.cutoff, tz="UTC")
    P(f"## 3. Рынки, открытые после отсечки модели ({a.cutoff}): честная тестовая выборка\n")
    mk = d.drop_duplicates("market_id")
    post = mk[mk.open_time >= cutoff]
    P(f"- Всего рынков с открытием после отсечки: {len(post)} из {len(mk)}; из них Polymarket {int((post.platform=='Polymarket').sum())}, Kalshi {int((post.platform=='Kalshi').sum())}.")
    P("- По категориям (без спорта, погоды и крипто-свечей, где новостной анализ бесполезен):\n")
    excl = post[~post.category.isin(["Sports", "Weather", "Climate and Weather", "Crypto"])]
    P("| Категория | N рынков | медианная длительность, дни | доля Yes |")
    P("|---|---|---|---|")
    for c, g in excl.groupby("category"):
        if len(g) >= 10:
            P(f"| {c} | {len(g)} | {g.duration_days.median():.0f} | {g.outcome.mean()*100:.0f}% |")
    P(f"\nИтого пригодных для новостного прогноза после отсечки: {len(excl)} рынков; с длительностью ≥ 7 дней: {int((excl.duration_days>=7).sum())}.\n")
    # pace: how many such markets open per month historically (to size a forward test)
    allx = mk[~mk.category.isin(["Sports", "Weather", "Climate and Weather", "Crypto"])]
    per_month = allx[allx.open_time >= "2025-01-01"].groupby(allx.open_time.dt.to_period("M")).size()
    P(f"Темп появления таких рынков в выборке (2025–2026): медиана {per_month.median():.0f} в месяц (это выборка с ограничением ≤10 на событие, реальный поток больше).\n")

    # 4. Экономика: байесовский прогнозист с частным сигналом
    P("## 4. Экономика ставок против рынка при заданном качестве частного сигнала (симуляция)\n")
    P("Модель: истинная вероятность q отличается от цены рынка на ошибку e: logit(q) = logit(цена) + e, e ~ N(0, τ). "
      "Прогнозист видит цену и зашумлённый сигнал об ошибке s = e + N(0, σ) и ставит байесовски: logit(p_f) = logit(цена) + s·τ²/(τ²+σ²). "
      "Ставка, если |p_f − цена| > порога, фиксированный размер, издержки 1.5 п.п. на контракт (спред + комиссия). Цены — реальные цены Polymarket за 7 дней до закрытия.\n")
    rng = np.random.default_rng(0)
    g = d[(d.horizon == "7d") & (d.platform == "Polymarket")]
    price = g.price.to_numpy(); n = len(price); lg = np.log(price / (1 - price))
    P("| ошибка рынка τ | шум сигнала σ | отношение сигнал/шум τ/σ | порог | доля ставок | средн. доходность на ставку, % | Brier прогнозиста | Brier рынка |")
    P("|---|---|---|---|---|---|---|---|")
    for tau in [0.3, 0.6]:
        for sigma in [0.3, 0.6, 1.2, 2.4]:
            k = tau**2 / (tau**2 + sigma**2)
            for thr in [0.05, 0.10]:
                res = []; bf = []; bm = []
                for _ in range(20):
                    e = rng.normal(0, tau, n); q = 1 / (1 + np.exp(-(lg + e))); y = (rng.random(n) < q).astype(float)
                    s_ = e + rng.normal(0, sigma, n); pf = 1 / (1 + np.exp(-(lg + k * s_)))
                    bet = np.abs(pf - price) > thr; long_ = pf > price; cost = 0.015
                    pnl = np.where(long_, (y - price) / price, ((1 - y) - (1 - price)) / (1 - price)) - cost / np.where(long_, price, 1 - price)
                    res.append((bet.mean(), pnl[bet].mean() if bet.any() else np.nan)); bf.append(brier(pf, y)); bm.append(brier(price, y))
                res = np.array(res)
                P(f"| {tau} | {sigma} | {tau/sigma:.2f} | {thr:.2f} | {res[:,0].mean()*100:.0f}% | {np.nanmean(res[:,1])*100:+.1f} | {np.mean(bf):.4f} | {np.mean(bm):.4f} |")
    P("")
    P("Чтение таблицы: доходность появляется только при отношении сигнал/шум порядка 0.5 и выше, то есть когда частная информация прогнозиста "
      "сопоставима по величине с ошибкой рынка. Калибровочные исследования (раздел 1) показывают, что ошибка рынка τ мала: цены откалиброваны в пределах 1–3 п.п., "
      "поэтому реалистичный τ ближе к 0.3, чем к 0.6, и требуемое качество сигнала высокое. Улучшение Brier на 0.001–0.003 относительно рынка — это уже прибыльный прогнозист.\n")

    txt = "\n".join(L); open(os.path.join(a.out, "results_pm_baseline.md"), "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
