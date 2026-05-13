import qlib

qlib.init(provider_uri="~/.qlib/qlib_data/qlib_bin")

from qlib.data import D

instruments = D.instruments()
# Extended fields: basic OHLCV + amount + derived indicators
# This gives LLM more dimensions to construct innovative factors
fields = [
    # Basic price & volume
    "$open", "$close", "$high", "$low", "$volume", "$factor",
    # Trading value
    "$amount",
    # Percentage change
    "$change",
    # Turnover rate (shares traded / total shares)
    "$turnover",
    # VWAP proxy (volume-weighted average price)
    # Note: not all qlib data bundles include $vwap; will be NaN if absent
    "$vwap",
]
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2008-12-29":].sort_index()

# Drop columns that are entirely NaN (fields not supported by this data bundle)
data = data.dropna(axis=1, how="all")

data.to_hdf("./daily_pv_all.h5", key="data")
debug_data = (
    D.features(instruments, fields, start_time="2018-01-01", end_time="2019-12-31", freq="day")
    .swaplevel()
    .sort_index()
)
# Drop columns that are entirely NaN (keep consistent with full data)
debug_data = debug_data.dropna(axis=1, how="all")

# Filter to first 100 instruments that exist in both full data and debug date range
all_instruments = data.reset_index()["instrument"].unique()
debug_instruments = debug_data.reset_index()["instrument"].unique()
common_instruments = [i for i in all_instruments[:200] if i in debug_instruments][:100]

debug_data = (
    debug_data
    .swaplevel()
    .loc[common_instruments]
    .swaplevel()
    .sort_index()
)

debug_data.to_hdf("./daily_pv_debug.h5", key="data")
