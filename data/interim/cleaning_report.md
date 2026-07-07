# Cleaning report

- Rows in (raw): **13348**
- Rows out (clean): **13003**
- Total dropped: **345** (2.6%)

| Reason | Rows dropped |
|---|---|
| Exact duplicate rows | 0 |
| Duplicate listing_key (kept freshest) | 0 |
| Missing required field | 317 |
| Price at/below $500 floor | 6 |
| Mileage outside 0–350,000 | 7 |
| Year outside 1990–next model year | 4 |
| Payment-amount artifact | 11 |

### Payment-amount artifacts caught (examples)

| Year | Make | Model | Mileage | Listed 'price' |
|---|---|---|---|---|
| 2025 | BMW | X5 | 22,154 | $999 |
| 2026 | Ford | Expedition | 2,552 | $1,550 |
| 2026 | GMC | Sierra | 706 | $1,850 |
| 2027 | Chevrolet | Corvette | 2,546 | $1,892 |
| 2027 | Chevrolet | Corvette | 2,546 | $1,938 |
| 2025 | Nissan | Kicks | 907 | $2,500 |
