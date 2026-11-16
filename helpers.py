from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas._typing import DropKeep


@dataclass(frozen=True)
class Columns:
    datetime: str = "datetime"
    symbol: str = "symbol"
    period: str = "period"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"

    @property
    def ohlc(self) -> list[str]:
        """Колонки цен OHLC."""
        return [self.open, self.high, self.low, self.close]

    @property
    def numeric(self) -> list[str]:
        """Числовые колонки."""
        return [*self.ohlc, self.volume]

    @property
    def required(self) -> list[str]:
        """Обязательные колонки сырых данных."""
        return [self.datetime, self.symbol, self.period, *self.ohlc, self.volume]

    @property
    def features(self) -> list[str]:
        """Колонки лог-отношений цен."""
        return [f"{column}_log_ratio" for column in self.ohlc]


def read_csv(
    path: Path,
    columns: Columns,
    *,
    sep: str = ",",
    encoding: str | None = None,
    datetime_format: str | None = None,
    mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    CSV с разделителем sep.
    mapping: {имя_в_файле: имя_атрибута_Layout}, например {"Date": "datetime"}.
    Если mapping не задан — колонки уже названы так, как в Layout.
    """
    df = pd.read_csv(path, sep=sep, encoding=encoding)

    if mapping:
        df = df.rename(
            columns={raw: getattr(columns, attr) for raw, attr in mapping.items()}
        )

    if columns.datetime not in df.columns:
        raise ValueError(f"В CSV отсутствует '{columns.datetime}'")

    df[columns.datetime] = pd.to_datetime(
        df[columns.datetime],
        format=datetime_format,
        errors="coerce",
    )
    return df


def read_txt(path: Path, columns: Columns) -> pd.DataFrame:
    """
    Фиксированный TXT-формат:
        TICKER PER DTYYYYMMDD TIME OPEN HIGH LOW CLOSE VOL OPENINT
    без строки заголовка (skiprows=1), DATE + TIME => одна колонка datetime.
    """
    df = pd.read_csv(
        path,
        header=None,
        skiprows=1,
        names=[
            "symbol",
            "period",
            "date",
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "openint",
        ],
        dtype={"date": str, "time": str},
    )
    date_str = df["date"].fillna("").astype(str)
    time_str = df["time"].fillna("").astype(str).str.zfill(6)
    df[columns.datetime] = pd.to_datetime(
        date_str + " " + time_str,
        format="%Y%m%d %H%M%S",
        errors="coerce",
    )
    return df


READERS: dict[str, Callable[[Path, Columns], pd.DataFrame]] = {
    ".csv": read_csv,
    ".txt": read_txt,
}


def load(
    path: Path | str,
    layout: Columns,
    *,
    readers: dict[str, Callable[[Path, Columns], pd.DataFrame]] = READERS,
) -> pd.DataFrame:
    """
    Читает файл выбранным reader'ом, проверяет схему, приводит типы,
    возвращает DataFrame строго с колонками layout.output.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")

    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Нет reader'а для {path.suffix}. Доступно: {list(readers)}")

    df = reader(path, layout)

    missing = [column for column in layout.required if column not in df.columns]
    if missing:
        raise KeyError(f"Отсутствуют колонки: {missing}")

    df[layout.numeric] = df[layout.numeric].apply(pd.to_numeric, errors="coerce")
    df[layout.numeric] = df[layout.numeric].replace([np.inf, -np.inf], np.nan)

    return df[layout.required]


def drop_invalid_datetime(df: pd.DataFrame, layout: Columns) -> pd.DataFrame:
    """Убирает строки, где datetime не распознался (NaT)."""
    return df.dropna(subset=[layout.datetime])


def sort_by_datetime(df: pd.DataFrame, layout: Columns) -> pd.DataFrame:
    """Сорировка по datetime"""
    return df.sort_values(layout.datetime, kind="mergesort")


def drop_nan_rows(df: pd.DataFrame, layout: Columns) -> pd.DataFrame:
    """Убирает строки с NaN в любой из колонок layout.output."""
    return df.dropna(subset=layout.required)


def drop_invalid_ohlc(df: pd.DataFrame, layout: Columns) -> pd.DataFrame:
    """Убирает строки с неверными OHLC."""
    prices = df[layout.ohlc]
    bad = (
        (prices[layout.high] < prices[layout.low])
        | (prices[layout.high] < prices[layout.open])
        | (prices[layout.high] < prices[layout.close])
        | (prices[layout.low] > prices[layout.open])
        | (prices[layout.low] > prices[layout.close])
    )
    return df[~bad]


def drop_non_positive_prices(df: pd.DataFrame, layout: Columns) -> pd.DataFrame:
    """Цены > 0. Обязательно, если дальше считать log-отношения."""
    return df[(df[layout.ohlc] > 0).all(axis=1)]


def drop_duplicate_datetime(
    df: pd.DataFrame,
    layout: Columns,
    *,
    keep: DropKeep = "last",
) -> pd.DataFrame:
    """
    Сортирует по datetime и удаляет дубли по datetime за один проход.
    """
    df = df.sort_values(layout.datetime, kind="mergesort")
    return df.drop_duplicates(subset=[layout.datetime], keep=keep)


def reset_index(df: pd.DataFrame, layout: Columns) -> pd.DataFrame:
    return df.reset_index(drop=True)


def run_steps(
    data: pd.DataFrame,
    steps: Sequence[Callable[[pd.DataFrame, Columns], pd.DataFrame]],
    columns: Columns,
) -> pd.DataFrame:
    """Применяет шаги по порядку и печатает, сколько строк удалил каждый шаг.

    Raises:
        KeyError: шагу не хватает колонок.
        ValueError: данные не проходят проверки шага.
    """
    print(f"{'step':25s} {'before':>10s} {'after':>10s} {'removed':>10s}")

    for step in steps:
        rows_before = len(data)
        data = step(data, columns)
        rows_removed = rows_before - len(data)
        print(
            f"{step.__name__:25s} {rows_before:>10_} {len(data):>10_} {rows_removed:>10_}"
        )

    return data


def to_kronos(df: pd.DataFrame, columns: Columns) -> pd.DataFrame:
    kronos_columns = ("timestamps", "open", "high", "low", "close", "volume", "amount")

    """
    Приводит df к формату Kronos.

    Kronos ожидает колонки:
        timestamps, open, high, low, close, volume, amount
    volume и amount опциональны в исходных данных — если их нет, ставим 0.
    """
    rename = {
        columns.datetime: "timestamps",
        columns.open: "open",
        columns.high: "high",
        columns.low: "low",
        columns.close: "close",
        columns.volume: "volume",
    }
    missing = [column for column in rename if column not in df.columns]
    if missing:
        raise KeyError(f"Для Kronos отсутствуют колонки: {missing}")

    result = df.rename(columns=rename)
    if "amount" not in result.columns:
        result = result.assign(amount=0.0)

    return result[list(kronos_columns)].reset_index(drop=True)
