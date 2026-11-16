import copy
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.path
import matplotlib.pyplot as plt
import pandas as pd
from model import Kronos, KronosPredictor, KronosTokenizer
from tqdm import tqdm

from helpers import (
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


def _safe_path_deepcopy(
    self: Path,
    memo: dict[int, Any] | None = None,
) -> Path:
    """Обход рекурсии Path.__deepcopy__ на Python 3.14 (matplotlib < 3.10.7)."""
    if memo is None:
        memo = {}
    cls = self.__class__
    new = cls.__new__(cls)
    memo[id(self)] = new
    for key, value in self.__dict__.items():
        setattr(new, key, copy.deepcopy(value, memo))
    return new


matplotlib.path.Path.__deepcopy__ = _safe_path_deepcopy  # type: ignore[method-assign]

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

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

tokenizer.eval()
model.eval()

predictor = KronosPredictor(
    model,
    tokenizer,
    device="cpu",
    max_context=512,
)


def rolling_forecast_one_bar(
    kronos_df: pd.DataFrame,
    predictor: KronosPredictor,
    lookback: int = 256,
    start_index: int | None = None,
    end_index: int | None = None,
) -> pd.DataFrame:
    """
    Для каждого t от lookback до конца данных:
      - контекст: [t-lookback : t]
      - прогноз: 1 бар в момент t
      - сохраняет реальный и прогнозный close.

    Возвращает DataFrame: timestamp, actual_close, predicted_close.
    """
    ohlcv_cols = ["open", "high", "low", "close", "volume", "amount"]

    # t — индекс первого прогнозируемого бара
    start = start_index if start_index is not None else lookback
    end = end_index if end_index is not None else len(kronos_df)

    results = []
    for t in tqdm(range(start, end), desc="Rolling forecast"):
        # Контекст: lookback баров ДО t
        x_df = kronos_df.iloc[t - lookback : t][ohlcv_cols].reset_index(drop=True)
        x_ts = kronos_df["timestamps"].iloc[t - lookback : t].reset_index(drop=True)
        y_ts = kronos_df["timestamps"].iloc[t : t + 1].reset_index(drop=True)

        pred = predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=1,
            T=2,
            top_p=1,
            sample_count=200,
            verbose=False,
        )

        results.append(
            {
                "timestamp": y_ts.iloc[0],
                "actual_close": float(kronos_df["close"].iloc[t]),
                "predicted_close": float(pred["close"].iloc[0]),
            }
        )

    return pd.DataFrame(results)

def rolling_forecast_one_bar(
    kronos_df: pd.DataFrame,
    predictor: KronosPredictor,
    lookback: int = 256,
    start_index: int | None = None,
    end_index: int | None = None,
) -> pd.DataFrame:
    """
    Для каждого t от lookback до конца данных:
      - контекст: [t-lookback : t]
      - прогноз: 1 бар в момент t
      - сохраняет реальный и прогнозный OHLC.

    Возвращает DataFrame:
        timestamp,
        actual_{open,high,low,close},
        predicted_{open,high,low,close}.
    """
    ohlcv_cols = ["open", "high", "low", "close", "volume", "amount"]
    ohlc_cols = ["open", "high", "low", "close"]

    start = start_index if start_index is not None else lookback
    end = end_index if end_index is not None else len(kronos_df)

    results = []
    for t in tqdm(range(start, end), desc="Rolling forecast"):
        x_df = kronos_df.iloc[t - lookback : t][ohlcv_cols].reset_index(drop=True)
        x_ts = kronos_df["timestamps"].iloc[t - lookback : t].reset_index(drop=True)
        y_ts = kronos_df["timestamps"].iloc[t : t + 1].reset_index(drop=True)

        pred = predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=1,
            T=2,
            top_p=1,
            sample_count=200,
            verbose=False,
        )

        row: dict = {"timestamp": y_ts.iloc[0]}
        for col in ohlc_cols:
            row[f"actual_{col}"] = float(kronos_df[col].iloc[t])
            row[f"predicted_{col}"] = float(pred[col].iloc[0])
        results.append(row)

    return pd.DataFrame(results)

def scatter_forecast_over_time(
    result: pd.DataFrame,
    *,
    out_path: Path | str,
    dpi: int = 150,
    max_points: int | None = None,
) -> Path:
    """
    Точечный график по времени: для каждого timestamp — две точки
    (actual_close и predicted_close). Линий нет.

    max_points — взять только последние N точек (для длинных рядов).
    """
    if result.empty:
        raise ValueError("Пустой результат прогноза")

    sub = result.sort_values("timestamp").reset_index(drop=True)
    if max_points is not None and len(sub) > max_points:
        sub = sub.iloc[-max_points:].reset_index(drop=True)

    ts = sub["timestamp"]
    actual = sub["actual_close"]
    predicted = sub["predicted_close"]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.scatter(
        ts, actual, s=14, color="black", alpha=0.65, label="Actual close", zorder=2
    )
    ax.scatter(
        ts,
        predicted,
        s=14,
        color="tab:orange",
        alpha=0.65,
        label="Predicted close (1-bar ahead)",
        zorder=3,
    )

    # # Опционально: вертикальные отрезки actual↔predicted — наглядно видно ошибку.
    for x, a, p in zip(ts, actual, predicted):
        ax.plot([x, x], [a, p], color="gray", linewidth=0.4, alpha=0.35, zorder=1)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Close")
    ax.set_title(f"Kronos 1-bar forecast over time (n={len(sub)})")
    ax.grid(alpha=0.3)
    ax.legend()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


result = rolling_forecast_one_bar(
    data,
    predictor,
    lookback=512,
    start_index=10_000,
    end_index=10_100,
)
print(result.head())
scatter_forecast_over_time(result, out_path="figures/time_h1.png")
