# AdEase — Wikipedia Page Views: EDA & Time Series Forecasting

Forecasting Wikipedia page-view traffic by language/region to help **AdEase** price and place ads more effectively. Built with **ARIMA**, **SARIMAX**, and **Facebook Prophet**.

## Business Context

AdEase places ads across Wikipedia pages in different languages. To price and place ads well, the business needs to **forecast page-view traffic by language/region** so clients know how their ads are likely to perform.

## Dataset

- **`train_1.csv`** — 69,709 Wikipedia pages × 550 days of daily views (2015-07-01 to 2016-12-31)
- **`Exog_Campaign_eng`** — a binary campaign/event flag for the English edition, used as an exogenous regressor

Each `Page` field is parsed (right-to-left, since article titles can contain underscores) into:
- `title` — article name
- `domain` / `language` — e.g. `en`, `ja`, `de`, `fr`, `zh`, `ru`, `es`, plus non-language projects `commons.wikimedia.org` and `www.mediawiki.org`
- `access` — `all-access` / `desktop` / `mobile-web`
- `agent` — `all-agents` / `spider` (bot traffic)

## What This Notebook Does

1. **Load & clean** the 145k-page, 550-day dataset plus the English campaign flag
2. **Parse** page names into title / language / access type / access agent
3. **Exploratory Data Analysis**
   - Missing-value analysis (~8.45% of cells null; treated as 0 views, pages >50% missing excluded from aggregates)
   - Total & average traffic by language
   - Daily aggregate trend per language
   - Day-of-week seasonality
   - Access-type and agent (human vs. spider/bot) traffic mix
   - Campaign/event effect on English traffic
4. **Build language-level aggregate series** (the unit AdEase needs for client-facing forecasts) and test **stationarity**:
   - Rolling mean/std
   - Augmented Dickey-Fuller (ADF) test
   - Seasonal decomposition (weekly, period=7)
   - Differencing (non-seasonal `d=1`, seasonal `D=1, s=7`)
   - ACF / PACF plots
5. **Build a reusable forecasting pipeline**:
   - **ARIMA** — non-seasonal baseline, orders chosen via `pmdarima.auto_arima` (AIC-minimizing grid search)
   - **SARIMAX** — seasonal (weekly, `m=7`), with the English campaign flag as an exogenous regressor
   - **Prophet** — weekly seasonality + optional campaign regressor
6. **Run the pipeline across all languages**, evaluate with **MAPE** on a 30-day holdout, and compare results
7. **Scale to individual pages** using the same reusable functions (demoed on top English pages)
8. **Business insights, recommendations, and case questionnaire answers**

## Key Results (30-day holdout MAPE, by language)

| Language | Best Model | MAPE |
|---|---|---|
| zh | SARIMAX | 4.67% |
| en | SARIMAX | 6.19% |
| fr | Prophet | 6.27% |
| commons | ARIMA | 7.88% |
| ja | Prophet | 8.70% |
| ru | Prophet | 11.60% |
| de | ARIMA | 14.24% |
| www | SARIMAX | 17.20% |
| es | SARIMAX | 88.46% |

*(`es` has very few pages in this sample and is an outlier — not representative of typical performance.)*

## Key Insights

- **English dominates traffic volume**, but that also implies the highest ad competition/CPM.
- **Every language shows weekly seasonality** — any model without a 7-day seasonal term (plain ARIMA) systematically mis-forecasts the weekday/weekend swing.
- **A meaningful share of traffic is bot/spider traffic**, not human — ad-exposure forecasts should be based on human ("all-agents") traffic, not raw totals.
- **Campaign/event days coincide with visible spikes** in English traffic, justifying the exogenous regressor in SARIMAX.
- **Forecast accuracy varies significantly by language** — higher-traffic, more stable editions forecast more reliably; low-traffic/volatile editions and non-language projects (`commons`, `www.mediawiki`) show higher error.

## Recommendations for AdEase

1. **Forecast at the language level** for pricing/planning; drop to page-level only for large, established pages.
2. **Use SARIMAX with the campaign flag for English** — it's the only model aware of promotional spikes.
3. **Use Prophet as the default for other languages** — handles weekly seasonality automatically with minimal per-language tuning, useful when scaling.
4. **Keep ARIMA only as a fast baseline/sanity-check**, not a production model — it structurally can't capture weekly seasonality.
5. **Report MAPE per language to clients**, not one blanket company-wide number.
6. **Separate spider/bot traffic from human traffic** when quoting expected ad impressions.

## Tech Stack

- Python, pandas, numpy, matplotlib, seaborn
- `statsmodels` (ADF test, seasonal decomposition, ACF/PACF, ARIMA, SARIMAX)
- `pmdarima` (`auto_arima` for automated order selection via AIC)
- `prophet` (Facebook/Meta's forecasting library)

## Repository Structure

```
.
├── AdEase_Wikipedia_Traffic_Forecasting_BusinessCase.ipynb   # Main analysis notebook
└── README.md
```


## How to Run

```bash
pip install pmdarima prophet pandas numpy matplotlib seaborn statsmodels
jupyter notebook AdEase_Wikipedia_Traffic_Forecasting_BusinessCase.ipynb
```


