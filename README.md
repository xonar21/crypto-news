# crypto-news

Исследование: есть ли устойчивая зависимость между выходом новостей и движением котировок, на которой можно построить торгового бота.

- `research/REPORT.md` — фаза 1: обзор литературы + событийный тест BTCUSDT вокруг макроновостей США
- `research/REPORT_PHASE2.md` — фаза 2: стратегии без гонки за скорость (техническое подтверждение, отложенные входы, стрэддл стоп-ордерами, дневные горизонты), BTC/ETH/SOL, внеплановые новости CryptoPanic
- `research/REPORT_PHASE3.md` — фаза 3: библиотека 48 кейсов внеплановых событий 2021–2026, категории, правило «уже в цене»
- `research/REPORT_PHASE4.md` — фаза 4: проверка правил на всех ценовых шоках, FOMC в деталях (заявление vs пресс-конференция), межрыночные связи, сезонность
- `research/event_study.py`, `research/phase2.py`, `research/robustness.py`, `research/placebo.py`, `research/case_library.py`, `research/phase4.py`, `research/phase4b.py` — код
- `research/results/` — таблицы и данные по событиям
