# How to read files.
For example, if you want to read `filename.h5`
```Python
import pandas as pd
df = pd.read_hdf("filename.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files **.

# Here is a short description about the data

| Filename       | Description                                                      |
| -------------- | -----------------------------------------------------------------|
| "daily_pv.h5"  | Forward-adjusted daily price, volume, and derived market data.          |


# For different data, We have some basic knowledge for them

## Daily price, volume and market data

**IMPORTANT**: All price columns (`$open`, `$close`, `$high`, `$low`) are already **forward-adjusted** for corporate actions (splits, dividends). Use them directly for returns, momentum, and any price-based calculations. Do NOT multiply by `$factor`.

### Basic Price Data
- `$open`: Forward-adjusted open price of the stock on that day.
- `$close`: Forward-adjusted close price of the stock on that day. Use directly — do NOT multiply by `$factor`.
- `$high`: Forward-adjusted high price of the stock on that day.
- `$low`: Forward-adjusted low price of the stock on that day.
- `$factor`: Cumulative adjustment factor. This is used internally by Qlib for backward-adjustment. You do NOT need to apply this to price data.

### Volume & Trading Data
- `$volume`: Volume (number of shares traded) of the stock on that day.
- `$amount`: Total trading value (in RMB) of the stock on that day. This is useful for constructing liquidity and money-flow based factors.

### Derived Price Data
- `$change`: Percentage change of the close price from the previous day: `(close_t - close_{t-1}) / close_{t-1}`.
- `$vwap`: Volume-weighted average price. A benchmark price that accounts for both price and volume distribution throughout the day.

**NOTE**: The available columns are ONLY: `$open`, `$close`, `$high`, `$low`, `$volume`, `$amount`, `$change`, `$vwap`, `$factor`. Do NOT use `$turnover` or any other column not listed here — they do NOT exist in the data.

### Suggested Factor Construction Directions
Using the above data dimensions, you can construct factors in categories such as:
1. **Momentum / Reversal**: Price changes over various windows using `$close`, `$change`.
2. **Volatility**: Price range (`$high - $low`), return volatility over windows.
3. **Volume-Price Divergence**: Discrepancy between `$volume`/`$amount` trends and price trends.
4. **Liquidity**: `$amount` based measures (e.g., Amihud illiquidity: `abs($change) / $amount`).
5. **Microstructure**: `$vwap` deviation from `$close`, intraday price distribution proxies.
6. **Cross-Sectional**: Rank-based or z-scored versions of the above across instruments.