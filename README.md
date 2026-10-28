# crypto-news

Исследование: есть ли устойчивая зависимость между выходом новостей и движением котировок, на которой можно построить торгового бота.

- `research/REPORT.md` — фаза 1: обзор литературы + событийный тест BTCUSDT вокруг макроновостей США
- `research/REPORT_PHASE2.md` — фаза 2: стратегии без гонки за скорость (техническое подтверждение, отложенные входы, стрэддл стоп-ордерами, дневные горизонты), BTC/ETH/SOL, внеплановые новости CryptoPanic
- `research/REPORT_PHASE3.md` — фаза 3: библиотека 48 кейсов внеплановых событий 2021–2026, категории, правило «уже в цене»
- `research/REPORT_PHASE4.md` — фаза 4: проверка правил на всех ценовых шоках, FOMC в деталях (заявление vs пресс-конференция), межрыночные связи, сезонность
- `research/REPORT_PHASE5.md` — фаза 5: позиционирование (поток ордеров) и реакция относительно ожидаемого
- `research/REPORT_STRATEGY.md` — итог: событийный портфель FOMC + CPI как единственный способ, прошедший все проверки; спецификация, устойчивость, план форвард-теста; сравнение с трендом, моментумом и парами на 15 монетах
- `research/REPORT_FOOTBALL.md` — футбол и ставки: проверка идеи «считать вероятность из параметров и ставить на наивысшую» на 12.7k матчей АПЛ с коэффициентами 24 букмекеров
- `research/REPORT_ARBITRAGE.md` — расхождения коэффициентов между конторами: вилки по сезонам и value по честной цене Pinnacle
- `research/PM_BACKTEST_DESIGN.md` — рынки предсказаний: как честно бэктестить ИИ-прогнозиста, планка на 46k рынков, план проверки
- `research/REPORT_TRADERS.md` — как торгуют реальные трейдеры: статистика прибыльности, источники преимущества, что есть у человека и чего нет у бота
- `research/BRAIN_DESIGN.md` — проектный документ: архитектура «трейдерского мозга» по итогам фаз 1–4
- `research/event_study.py`, `research/phase2.py`, `research/robustness.py`, `research/placebo.py`, `research/case_library.py`, `research/phase4.py`, `research/phase4b.py`, `research/phase5.py`, `research/phase5b.py`, `research/pm_baseline.py`, `research/football_analysis.py`, `research/arbitrage_analysis.py`, `research/event_portfolio.py`, `research/event_portfolio_robust.py`, `research/trend_pairs.py` — код
- `research/results/` — таблицы и данные по событиям
