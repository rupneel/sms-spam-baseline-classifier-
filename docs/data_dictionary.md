# Data Dictionary — SMS Spam Collection (UCI)

## Source
- **Origin**: UCI Machine Learning Repository
- **Collected by**: Tiago A. Almeida & José María Gómez Hidalgo
- **Period**: 2009–2012
- **URL**: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

## Schema

| Field     | Type   | Description                           | Values / Units         | Grain   |
|-----------|--------|---------------------------------------|------------------------|---------|
| `label`   | string | Classification of the SMS message     | `ham` (legitimate) or `spam` | per SMS |
| `message` | string | Raw text content of the SMS           | Free text, max ~910 chars    | per SMS |

## Key Facts
- **Row count**: 5,574 messages (expected)
- **Class balance**: ~87% ham, ~13% spam (imbalanced)
- **Language**: English only
- **Encoding**: Latin-1 (ISO 8859-1) in the raw file

## Known Caveats
1. **English-only** — results do not generalise to other languages.
2. **Dated collection** (2009–2012) — modern spam patterns may differ.
3. **Class imbalance** — majority-class baseline achieves ~87% accuracy by
   predicting "ham" for every message.
4. **No sender metadata** — only message text is available; no phone numbers,
   timestamps, or carrier info.
5. **Duplicate messages** — the raw file contains exact-duplicate rows that
   must be removed to avoid data leakage.
