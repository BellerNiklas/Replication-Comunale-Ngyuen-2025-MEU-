# Replication of Comunale & Nguyen (2025): MacroEconomic Uncertainty for the Euro Area

## Project Overview

This project replicates the MacroEconomic Uncertainty (MEU) measure for the euro area developed by Comunale and Nguyen (2025, Journal of International Money and Finance). The MEU follows the methodology of Jurado, Ludvigson, and Ng (AER 2015), measuring uncertainty as the conditional volatility of unforecastable components of macroeconomic time series.

## Data

The replication requires a large monthly dataset of approximately **1,330 variables** (after cleaning) covering **19 euro area countries** from **January 2003** onward.

### Data Sources
- **Eurostat**: Industrial production, labor market, prices, trade, sentiment indicators
- **ECB Statistical Data Warehouse (SDW)**: Financial variables, monetary aggregates, exchange rates
- **OECD**: Confidence indicators, leading indicators, long-term rates, share prices
- **BIS**: Nominal effective exchange rates

### Country-Specific Variables (up to 122 per country)

| Category | Examples |
|----------|----------|
| Industrial Production | Total industry, manufacturing, capital goods, consumer goods, energy |
| Labor Market | Employment indices, unemployment rates, hours worked, wages |
| Prices | PPI, HICP (overall, energy, food, services), import price indices |
| Activity Indicators | Car registrations, turnover indices, building permits |
| Trade | Imports and exports with world |
| Sentiment & Surveys | Economic sentiment, consumer/industrial/services confidence |
| Financial | Loans, deposits, debt securities, share prices, spreads |

### Euro Area-Level Variables (30 series)
- Government bond yields (2y, 3y, 5y, 7y, 10y)
- Money market rates (Euribor 1m, 3m, 6m, 1y; Eonia)
- Dow Jones Euro Stoxx indices (broad and sector-specific)
- Monetary aggregates (M1, M3, currency in circulation)
- Bilateral exchange rates (USD, GBP, JPY, CHF, CNY)

## References

- Jurado, K., Ludvigson, S.C., & Ng, S. (2015). Measuring Uncertainty. *American Economic Review*, 105(3), 1177-1216.
- Comunale, M., & Nguyen, A.D.M. (2025). A comprehensive MacroEconomic uncertainty measure for the euro area. *Journal of International Money and Finance*, 157, 103370.

