"""
Фаза 3: библиотека кейсов — крупные внеплановые события 2021-2026 и путь цены BTC/ETH/SOL вокруг них.

Для каждого события задано окно поиска (UTC); момент «рынок узнал» (onset) определяется автоматически как начало
5-минутного интервала с максимальным |доходность| внутри окна. Где известно точное время публикации, оно указано
отдельно (known), чтобы сравнить скорость реакции.

Запуск: python3 research/case_library.py --btc DIR --eth DIR --sol DIR --out research/results
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy import stats
from phase2 import Px

# category: SHOCK (внезапный негатив/позитив без предварительной подготовки), RUMOR (слух/первый сигнал цепочки),
# EXPECTED (запланированное бинарное событие, вероятность которого рынок уже оценивал), FAKE (ложная новость),
# REG (регуляторное решение с медленной имплементацией), GEO (геополитика), MACRO (макро-шок вне календаря),
# RELIEF (снятие риска: пауза, перемирие, спасение)
# exp: ожидаемое направление для BTC по логической цепочке (+1/-1). priced: что было заложено до события.
EVENTS = [
    dict(d="2021-02-08", name="Tesla купила BTC на $1.5B (8-K)", cat="SHOCK", exp=+1, known="2021-02-08 12:05", win=("2021-02-08 10:00", "2021-02-08 16:00"), priced="Слухи после твитов Маска в январе; факт неожиданный"),
    dict(d="2021-05-12", name="Маск: Tesla прекращает приём BTC", cat="SHOCK", exp=-1, known="2021-05-12 22:06", win=("2021-05-12 21:00", "2021-05-13 02:00"), priced="Неожиданно"),
    dict(d="2021-05-19", name="Китай: запрет финучреждениям на крипто-операции (крах 19 мая)", cat="REG", exp=-1, known=None, win=("2021-05-18 20:00", "2021-05-19 16:00"), priced="Заявление 18 мая, каскад ликвидаций 19 мая"),
    dict(d="2021-06-09", name="Сальвадор: BTC законное платёжное средство", cat="REG", exp=+1, known="2021-06-09 05:00", win=("2021-06-09 03:00", "2021-06-09 09:00"), priced="Анонс 5 июня на конференции, голосование ожидалось"),
    dict(d="2021-09-24", name="Китай: все крипто-транзакции незаконны (НБК)", cat="REG", exp=-1, known="2021-09-24 09:00", win=("2021-09-24 07:00", "2021-09-24 14:00"), priced="Повтор прежних запретов, но жёстче"),
    dict(d="2022-02-24", name="Вторжение России в Украину", cat="GEO", exp=-1, known="2022-02-24 03:00", win=("2022-02-24 01:00", "2022-02-24 08:00"), priced="Напряжение нарастало недели, рынок частично заложил"),
    dict(d="2022-05-09", name="Депег UST / крах Terra (основной день)", cat="SHOCK", exp=-1, known=None, win=("2022-05-09 00:00", "2022-05-10 00:00"), priced="Первый депег 7-8 мая, паника 9-го"),
    dict(d="2022-06-13", name="Celsius останавливает вывод средств", cat="SHOCK", exp=-1, known="2022-06-13 00:10", win=("2022-06-12 22:00", "2022-06-13 06:00"), priced="Слухи о проблемах Celsius неделю"),
    dict(d="2022-11-06", name="CZ: Binance продаёт FTT (начало краха FTX)", cat="RUMOR", exp=-1, known="2022-11-06 15:47", win=("2022-11-06 14:00", "2022-11-06 20:00"), priced="Статья CoinDesk 2 ноября, рынок не отреагировал"),
    dict(d="2022-11-08", name="FTX останавливает вывод, Binance LOI", cat="SHOCK", exp=-1, known="2022-11-08 15:00", win=("2022-11-08 10:00", "2022-11-08 18:00"), priced="Отток $6B за 72ч был виден on-chain"),
    dict(d="2022-11-09", name="Binance отказывается от покупки FTX", cat="SHOCK", exp=-1, known="2022-11-09 21:00", win=("2022-11-09 18:00", "2022-11-10 00:00"), priced="Сделка считалась спасением"),
    dict(d="2023-03-11", name="USDC депег после краха SVB (Circle $3.3B в SVB)", cat="SHOCK", exp=-1, known="2023-03-11 04:00", win=("2023-03-10 22:00", "2023-03-11 10:00"), priced="SVB закрыт днём 10 марта"),
    dict(d="2023-03-12", name="ФРС/Минфин/FDIC гарантируют депозиты SVB", cat="RELIEF", exp=+1, known="2023-03-12 22:15", win=("2023-03-12 20:00", "2023-03-13 04:00"), priced="Ожидалось решение до открытия понедельника"),
    dict(d="2023-06-05", name="SEC подаёт иск против Binance", cat="REG", exp=-1, known="2023-06-05 14:00", win=("2023-06-05 12:00", "2023-06-05 18:00"), priced="Расследование известно, иск неожиданный по времени"),
    dict(d="2023-06-06", name="SEC подаёт иск против Coinbase", cat="REG", exp=-1, known="2023-06-06 12:30", win=("2023-06-06 11:00", "2023-06-06 16:00"), priced="Wells notice в марте, ожидалось"),
    dict(d="2023-07-13", name="Суд: XRP на биржах не ценная бумага (Ripple vs SEC)", cat="REG", exp=+1, known="2023-07-13 17:50", win=("2023-07-13 16:00", "2023-07-13 21:00"), priced="Решение ожидалось, исход неизвестен"),
    dict(d="2023-08-29", name="Суд: SEC неправомерно отклонила Grayscale ETF", cat="REG", exp=+1, known="2023-08-29 14:00", win=("2023-08-29 12:00", "2023-08-29 18:00"), priced="Решение ожидалось, дата неизвестна"),
    dict(d="2023-10-16", name="Cointelegraph: ложное «SEC одобрила ETF BlackRock»", cat="FAKE", exp=+1, known="2023-10-16 13:24", win=("2023-10-16 12:00", "2023-10-16 16:00"), priced="Ожидание одобрения ETF росло"),
    dict(d="2023-10-23", name="iShares ETF появился в списке DTCC, пробой $30k", cat="RUMOR", exp=+1, known=None, win=("2023-10-23 10:00", "2023-10-24 02:00"), priced="Вероятность одобрения ETF ~65-90% по Bloomberg"),
    dict(d="2023-11-21", name="CZ признаёт вину, уходит из Binance, штраф $4.3B", cat="REG", exp=-1, known="2023-11-21 19:30", win=("2023-11-21 15:00", "2023-11-22 00:00"), priced="Слухи о сделке с DOJ за день"),
    dict(d="2024-01-09", name="Взлом аккаунта SEC в X: ложное одобрение ETF", cat="FAKE", exp=+1, known="2024-01-09 21:11", win=("2024-01-09 20:30", "2024-01-10 00:00"), priced="Одобрение ожидалось на следующий день (Polymarket ~88%)"),
    dict(d="2024-01-10", name="SEC одобряет спотовые BTC ETF", cat="EXPECTED", exp=+1, known="2024-01-10 21:00", win=("2024-01-10 19:00", "2024-01-11 02:00"), priced="Polymarket 85-89%, Bloomberg 95%"),
    dict(d="2024-04-13", name="Иран атакует Израиль дронами и ракетами", cat="GEO", exp=-1, known="2024-04-13 20:00", win=("2024-04-13 17:00", "2024-04-14 02:00"), priced="Атака ожидалась несколько дней"),
    dict(d="2024-05-20", name="Bloomberg поднимает шансы ETH ETF с 25% до 75%", cat="RUMOR", exp=+1, known="2024-05-20 17:00", win=("2024-05-20 15:00", "2024-05-20 21:00"), priced="Рынок ждал отказа (25%)"),
    dict(d="2024-05-23", name="SEC одобряет ETH ETF (19b-4)", cat="EXPECTED", exp=+1, known="2024-05-23 20:30", win=("2024-05-23 18:00", "2024-05-24 02:00"), priced="После 20 мая ~75%+, цена уже выросла на 20%"),
    dict(d="2024-07-05", name="Mt. Gox начинает выплаты (перевод $2.7B)", cat="SHOCK", exp=-1, known=None, win=("2024-07-04 20:00", "2024-07-05 08:00"), priced="Дата выплат «в начале июля» объявлена 24 июня"),
    dict(d="2024-08-05", name="Крах carry-trade (Nikkei −12%), BTC до $49k", cat="MACRO", exp=-1, known=None, win=("2024-08-04 18:00", "2024-08-05 08:00"), priced="Распродажа началась в пятницу 2 августа"),
    dict(d="2024-11-06", name="Победа Трампа на выборах", cat="EXPECTED", exp=+1, known=None, win=("2024-11-06 00:00", "2024-11-06 12:00"), priced="Polymarket ~58-62% за Трампа накануне"),
    dict(d="2025-01-27", name="DeepSeek: распродажа AI/рисковых активов", cat="MACRO", exp=-1, known=None, win=("2025-01-26 20:00", "2025-01-27 16:00"), priced="Неожиданно"),
    dict(d="2025-02-03", name="Тарифы Трампа на Канаду/Мексику (крах выходных)", cat="MACRO", exp=-1, known=None, win=("2025-02-02 18:00", "2025-02-03 08:00"), priced="Указ подписан 1 февраля (суббота)"),
    dict(d="2025-02-21", name="Взлом Bybit на $1.46B", cat="SHOCK", exp=-1, known="2025-02-21 14:16", win=("2025-02-21 13:00", "2025-02-21 20:00"), priced="Неожиданно"),
    dict(d="2025-03-02", name="Трамп: стратегический крипторезерв (XRP, SOL, ADA, потом BTC, ETH)", cat="SHOCK", exp=+1, known="2025-03-02 15:30", win=("2025-03-02 14:00", "2025-03-02 20:00"), priced="Резерв обсуждался с января, состав неожиданный"),
    dict(d="2025-03-07", name="Указ о стратегическом резерве BTC (без покупок) — «продай на факте»", cat="EXPECTED", exp=+1, known="2025-03-07 01:00", win=("2025-03-06 23:00", "2025-03-07 06:00"), priced="Анонс 2 марта, рынок ждал покупок"),
    dict(d="2025-04-02", name="«День освобождения»: глобальные тарифы", cat="MACRO", exp=-1, known="2025-04-02 20:15", win=("2025-04-02 19:00", "2025-04-03 02:00"), priced="Тарифы ожидались, масштаб нет"),
    dict(d="2025-04-09", name="Трамп: пауза тарифов на 90 дней", cat="RELIEF", exp=+1, known="2025-04-09 17:18", win=("2025-04-09 16:00", "2025-04-09 21:00"), priced="Неожиданно"),
    dict(d="2025-05-12", name="США и Китай: снижение тарифов (Женева)", cat="RELIEF", exp=+1, known="2025-05-12 07:00", win=("2025-05-12 05:00", "2025-05-12 14:00"), priced="Переговоры известны, результат неожиданно сильный"),
    dict(d="2025-06-13", name="Израиль наносит удары по Ирану", cat="GEO", exp=-1, known="2025-06-13 00:00", win=("2025-06-12 22:00", "2025-06-13 06:00"), priced="Напряжение известно, время нет"),
    dict(d="2025-06-21", name="США наносят удары по ядерным объектам Ирана", cat="GEO", exp=-1, known="2025-06-21 23:50", win=("2025-06-21 22:00", "2025-06-22 06:00"), priced="Трамп дал «две недели» 19 июня"),
    dict(d="2025-06-23", name="Трамп объявляет перемирие Израиль-Иран", cat="RELIEF", exp=+1, known="2025-06-23 22:05", win=("2025-06-23 20:00", "2025-06-24 04:00"), priced="Иран ответил символически, рынок уже разворачивался"),
    dict(d="2025-07-18", name="Подписание GENIUS Act (стейблкоины)", cat="EXPECTED", exp=+1, known="2025-07-18 20:30", win=("2025-07-18 18:00", "2025-07-19 02:00"), priced="Полностью ожидаемо после Сената 68-30"),
    dict(d="2025-10-10", name="Трамп: 100% тарифы на Китай → каскад ликвидаций $19B", cat="MACRO", exp=-1, known="2025-10-10 14:57", win=("2025-10-10 14:00", "2025-10-11 00:00"), priced="Неожиданно"),
    dict(d="2026-01-30", name="Трамп номинирует Уорша главой ФРС", cat="MACRO", exp=-1, known=None, win=("2026-01-30 14:00", "2026-01-31 08:00"), priced="Уорш был фаворитом, но считался про-крипто"),
    dict(d="2026-02-28", name="США и Израиль начинают удары по Ирану (война 2026)", cat="GEO", exp=-1, known=None, win=("2026-02-28 00:00", "2026-03-01 00:00"), priced="Переговоры шли, удар неожиданный по времени"),
    dict(d="2026-04-08", name="Трамп объявляет двухнедельное перемирие с Ираном", cat="RELIEF", exp=+1, known=None, win=("2026-04-07 20:00", "2026-04-08 08:00"), priced="За 10 часов до этого угроза «финального удара»"),
    dict(d="2026-04-12", name="Трамп: блокада Ормузского пролива флотом США", cat="GEO", exp=-1, known="2026-04-12 14:00", win=("2026-04-12 12:00", "2026-04-12 20:00"), priced="Переговоры в Пакистане провалились накануне"),
    dict(d="2026-05-14", name="CLARITY Act одобрен банковским комитетом Сената 15-9", cat="REG", exp=+1, known=None, win=("2026-05-14 13:00", "2026-05-14 22:00"), priced="Polymarket 67% на принятие в 2026"),
    dict(d="2026-07-12", name="Трамп восстанавливает блокаду Ирана (Brent +9.6%)", cat="GEO", exp=-1, known=None, win=("2026-07-12 12:00", "2026-07-13 06:00"), priced="Перемирие уже трещало"),
    dict(d="2026-09-15", name="Сенат проваливает cloture по CLARITY Act (49-50)", cat="REG", exp=-1, known=None, win=("2026-09-15 18:00", "2026-09-16 04:00"), priced="Polymarket 31-47% летом, исход неопределённый"),
]

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
H = [15, 60, 240, 1440, 4320, 10080]  # 15м, 1ч, 4ч, 1д, 3д, 7д


def detect_onset(px: Px, w0, w1, sign):
    """Начало 5-минутного окна с максимальным движением в ожидаемом направлении (sign) внутри окна поиска."""
    i, j = px.idx(pd.Timestamp(w0, tz="UTC")), px.idx(pd.Timestamp(w1, tz="UTC"))
    if j - i < 6:
        return None
    o = px.o[i:j]
    r5 = o[5:] / o[:-5] - 1
    k = int(np.argmax(sign * r5))
    return pd.Timestamp(px.ts[i + k] * 60, unit="s", tz="UTC"), r5[k]


def analyze(px_by_sym):
    rows = []
    for e in EVENTS:
        onset, r5 = detect_onset(px_by_sym["BTCUSDT"], *e["win"], e["exp"])
        if onset is None:
            continue
        # t0 = момент, когда трейдер мог узнать: известное время публикации, иначе детектированный onset
        t0 = pd.Timestamp(e["known"], tz="UTC") if e["known"] else onset
        rec = dict(date=e["d"], name=e["name"], category=e["cat"], exp=e["exp"], known=e["known"], onset=t0, r5_onset=r5, priced=e["priced"])
        if e["known"]:
            rec["lag_min"] = (onset - pd.Timestamp(e["known"], tz="UTC")).total_seconds() / 60
        else:
            rec["lag_min"] = np.nan
        for s in SYMS:
            px = px_by_sym[s]
            rec[f"{s}_pre7d"] = px.ret(t0, -10080, 0)
            rec[f"{s}_pre24h"] = px.ret(t0, -1440, 0)
            for k in H:
                rec[f"{s}_r{k}"] = px.ret(t0, 0, k)
        rows.append(rec)
    return pd.DataFrame(rows)


def fmt(x):
    return "n/a" if pd.isna(x) else f"{x*100:+.1f}"


def report(df, out_dir):
    L = []; P = L.append
    P("# Библиотека кейсов: 48 крупных внеплановых событий 2021–2026\n")
    P("Точка отсчёта t0 = известное время публикации; если оно неизвестно, то onset = начало 5-минутного интервала с максимальным движением BTC в ожидаемом направлении внутри окна поиска. "
      "Lag = onset минус время публикации (минуты; отрицательное = основное движение началось до публикации). "
      "Доходности BTC от t0, %. exp = ожидаемое направление по логической цепочке.\n")
    P("| Дата | Событие | Кат. | exp | lag, мин | pre 7д | pre 24ч | +15м | +1ч | +4ч | +1д | +3д | +7д |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        lag = "—" if pd.isna(r.lag_min) else f"{r.lag_min:+.0f}"
        P(f"| {r.date} | {r['name']} | {r.category} | {'+' if r.exp>0 else '−'} | {lag} | {fmt(r.BTCUSDT_pre7d)} | {fmt(r.BTCUSDT_pre24h)} | "
          f"{fmt(r.BTCUSDT_r15)} | {fmt(r.BTCUSDT_r60)} | {fmt(r.BTCUSDT_r240)} | {fmt(r.BTCUSDT_r1440)} | {fmt(r.BTCUSDT_r4320)} | {fmt(r.BTCUSDT_r10080)} |")
    P("")

    P("## Сводка по всем событиям (BTC)\n")
    b = df
    d1 = np.sign(b.BTCUSDT_r60)
    P(f"- Событий: {len(b)}. Направление первого часа совпало с ожидаемым по логике (exp): {(d1 == b.exp).mean()*100:.0f}%.")
    P(f"- Направление первых суток совпало с exp: {(np.sign(b.BTCUSDT_r1440) == b.exp).mean()*100:.0f}%; первых 7 дней: {(np.sign(b.BTCUSDT_r10080) == b.exp).mean()*100:.0f}%.")
    P(f"- Медиана |r| BTC: 15м {b.BTCUSDT_r15.abs().median()*100:.1f}%, 1ч {b.BTCUSDT_r60.abs().median()*100:.1f}%, 4ч {b.BTCUSDT_r240.abs().median()*100:.1f}%, "
      f"1д {b.BTCUSDT_r1440.abs().median()*100:.1f}%, 3д {b.BTCUSDT_r4320.abs().median()*100:.1f}%, 7д {b.BTCUSDT_r10080.abs().median()*100:.1f}%.")
    share = lambda k: (b[f"BTCUSDT_r{k}"] / b.BTCUSDT_r1440).clip(-3, 3).median()
    P(f"- Доля суточного движения, реализованная к +15м: {share(15)*100:.0f}%, к +1ч: {share(60)*100:.0f}%, к +4ч: {share(240)*100:.0f}% (медианы).")
    P(f"- Знак +1ч совпал со знаком +1д: {(np.sign(b.BTCUSDT_r60)==np.sign(b.BTCUSDT_r1440)).mean()*100:.0f}%; знак +1д со знаком +7д: {(np.sign(b.BTCUSDT_r1440)==np.sign(b.BTCUSDT_r10080)).mean()*100:.0f}%.")
    known = b.dropna(subset=["lag_min"])
    P(f"- Для {len(known)} событий с известным временем публикации: медианный lag onset {known.lag_min.median():+.0f} мин, "
      f"в {(known.lag_min < 0).mean()*100:.0f}% случаев основное движение началось ДО публикации.")
    P("")

    P("## Отложенные входы на событиях (без гонки): вход через 1 час по направлению первого часа\n")
    P("| Выборка | N | выход +4ч: средн. % / t / hit | выход +1д: средн. % / t / hit | выход +3д | выход +7д |")
    P("|---|---|---|---|---|---|")
    def cont(sub, s="BTCUSDT"):
        d = np.sign(sub[f"{s}_r60"])
        cells = []
        for k in [240, 1440, 4320, 10080]:
            g = d * ((1 + sub[f"{s}_r{k}"]) / (1 + sub[f"{s}_r60"]) - 1)
            g = g.dropna()
            t = stats.ttest_1samp(g, 0).statistic if len(g) >= 3 else np.nan
            cells.append(f"{g.mean()*100:+.2f} / {t:+.1f} / {(g>0).mean()*100:.0f}%")
        return cells
    for lab, sub in [("все, BTC", b), ("все, ETH", b), ("все, SOL", b)] + [(f"{c}, BTC", b[b.category == c]) for c in sorted(b.category.unique())]:
        s = "ETHUSDT" if "ETH" in lab else "SOLUSDT" if "SOL" in lab else "BTCUSDT"
        if len(sub) < 3:
            continue
        P(f"| {lab} | {len(sub)} | " + " | ".join(cont(sub, s)) + " |")
    P("")

    P("## Вход через сутки по направлению первых суток (дрейф), выход +3д / +7д\n")
    P("| Выборка | N | выход +3д: средн. % / t / hit | выход +7д: средн. % / t / hit |")
    P("|---|---|---|---|")
    for lab, sub in [("все", b)] + [(c, b[b.category == c]) for c in sorted(b.category.unique())]:
        if len(sub) < 3:
            continue
        d = np.sign(sub.BTCUSDT_r1440); cells = []
        for k in [4320, 10080]:
            g = (d * ((1 + sub[f"BTCUSDT_r{k}"]) / (1 + sub.BTCUSDT_r1440) - 1)).dropna()
            t = stats.ttest_1samp(g, 0).statistic if len(g) >= 3 else np.nan
            cells.append(f"{g.mean()*100:+.2f} / {t:+.1f} / {(g>0).mean()*100:.0f}%")
        P(f"| {lab} | {len(sub)} | " + " | ".join(cells) + " |")
    P("")

    P("## По категориям: величина и скорость (BTC, медианы)\n")
    P("| Кат. | N | \\|r1ч\\| | \\|r1д\\| | \\|r7д\\| | доля 1д к +1ч | 1ч→1д тот же знак | 1д→7д тот же знак | 1ч совпал с exp | 7д совпал с exp |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for c, g in b.groupby("category"):
        P(f"| {c} | {len(g)} | {g.BTCUSDT_r60.abs().median()*100:.1f}% | {g.BTCUSDT_r1440.abs().median()*100:.1f}% | {g.BTCUSDT_r10080.abs().median()*100:.1f}% | "
          f"{(g.BTCUSDT_r60/g.BTCUSDT_r1440).clip(-3,3).median()*100:.0f}% | {(np.sign(g.BTCUSDT_r60)==np.sign(g.BTCUSDT_r1440)).mean()*100:.0f}% | "
          f"{(np.sign(g.BTCUSDT_r1440)==np.sign(g.BTCUSDT_r10080)).mean()*100:.0f}% | {(np.sign(g.BTCUSDT_r60)==g.exp).mean()*100:.0f}% | {(np.sign(g.BTCUSDT_r10080)==g.exp).mean()*100:.0f}% |")
    P("")

    P("## «Уже в цене?»: зависит ли исход от движения за 7 дней ДО события (BTC)\n")
    P("Сделка по логике: вход в t0+60 мин в ожидаемом направлении exp, выход +1д / +7д. Разбивка по тому, шла ли цена в сторону exp уже до события.\n")
    P("| Группа | N | +1д: средн. % / t / hit | +7д: средн. % / t / hit |")
    P("|---|---|---|---|")
    pre_al = b.exp * b.BTCUSDT_pre7d
    for lab, m in [("все события", pre_al.notna()), ("цена уже прошла > 5% в сторону exp за 7 дней до", pre_al > 0.05),
                   ("цена прошла 0–5% в сторону exp", (pre_al > 0) & (pre_al <= 0.05)), ("цена шла ПРОТИВ exp до события", pre_al < 0)]:
        sub = b[m]; cells = []
        for k in [1440, 10080]:
            g = (sub.exp * ((1 + sub[f"BTCUSDT_r{k}"]) / (1 + sub.BTCUSDT_r60) - 1)).dropna()
            t = stats.ttest_1samp(g, 0).statistic if len(g) >= 3 else np.nan
            cells.append(f"{g.mean()*100:+.2f} / {t:+.1f} / {(g>0).mean()*100:.0f}%")
        P(f"| {lab} | {len(sub)} | " + " | ".join(cells) + " |")
    P("")
    P("Противоположная сделка (fade): если цена уже прошла > 5% в сторону exp за 7 дней, войти в t0+60 ПРОТИВ exp:\n")
    sub = b[pre_al > 0.05]
    for k in [1440, 4320, 10080]:
        g = (-sub.exp * ((1 + sub[f"BTCUSDT_r{k}"]) / (1 + sub.BTCUSDT_r60) - 1)).dropna()
        t = stats.ttest_1samp(g, 0).statistic
        P(f"- выход +{k//1440}д: N={len(g)}, средн. {g.mean()*100:+.2f}%, t={t:+.1f}, hit {(g>0).mean()*100:.0f}%")
    P("")
    P("## ETH и SOL: отношение движения к BTC (бета на событиях)\n")
    for s in ["ETHUSDT", "SOLUSDT"]:
        for k in [60, 1440]:
            ratio = (b[f"{s}_r{k}"] / b[f"BTCUSDT_r{k}"]).replace([np.inf, -np.inf], np.nan)
            same = (np.sign(b[f"{s}_r{k}"]) == np.sign(b[f"BTCUSDT_r{k}"])).mean()
            P(f"- {s}, горизонт {k} мин: медианное отношение движения к BTC {ratio.median():.2f}, совпадение знака {same*100:.0f}%")
    P("")

    txt = "\n".join(L)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "results_cases.md"), "w").write(txt)
    df.to_csv(os.path.join(out_dir, "cases.csv"), index=False)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc", required=True); ap.add_argument("--eth", required=True); ap.add_argument("--sol", required=True)
    ap.add_argument("--out", default="research/results")
    a = ap.parse_args()
    px = {"BTCUSDT": Px(a.btc, "BTCUSDT"), "ETHUSDT": Px(a.eth, "ETHUSDT"), "SOLUSDT": Px(a.sol, "SOLUSDT")}
    df = analyze(px)
    report(df, a.out)


if __name__ == "__main__":
    main()
