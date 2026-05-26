# Housing Market Screener

This script ranks US housing/rental markets using:

- 3-month, 6-month, and 9-month moving averages
- First derivative: month-over-month home value change
- Second derivative: acceleration in monthly home value change
- Rent growth
- Gross rent yield proxy
- Combined pickup/investment score

## Install

```bash
pip install pandas numpy
```

## Download data

Download Zillow Research CSV files manually:

- ZHVI home value file, for example: `Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv`
- ZORI rent file, for example: `Metro_zori_sm_month.csv`

Place both files in the same folder as `housing_market_screener.py`, or pass their full paths.

## Run

```bash
python housing_market_screener.py \
  --home-file Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv \
  --rent-file Metro_zori_sm_month.csv \
  --output-file top_markets.csv \
  --top-n 25
```

The output file `top_markets.csv` will contain the ranked markets.
