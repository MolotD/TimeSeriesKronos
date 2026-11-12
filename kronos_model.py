from load_helpers import (
    Columns,
    drop_duplicate_datetime,
    drop_invalid_datetime,
    drop_invalid_ohlc,
    drop_nan_rows,
    drop_non_positive_prices,
    load,
    reset_index,
    run_steps,
    sort_by_datetime,
    to_kronos,
)

columns = Columns()
data = load("/mnt/e/TradingViewData/SPX 1D_20_000.csv", columns)

steps = [
    drop_invalid_datetime,
    sort_by_datetime,
    drop_nan_rows,
    drop_invalid_ohlc,
    drop_non_positive_prices,
    drop_duplicate_datetime,
    reset_index,
]

data = run_steps(data, steps, columns)
data = to_kronos(data, columns)
print(data.tail())
