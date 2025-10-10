
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DA-3-29 — Создание признаков: скользящее окно с агрегацией
---------------------------------------------------------

Функциональность:
1) Загружает Iris через функции из DA_1_07.py и добавляет категориальный признак species.
2) Для выбранного числового признака считает rolling-статистики в окне N: mean, min, max, std
   отдельно внутри каждой группы (species).
3) Добавляет эти статистики как новые признаки.
4) Строит 4 графика (динамика по индексам внутри каждой группы).
5) Сохраняет итоговый DataFrame в CSV.

Пример запуска:
    python3 DA-3-29.py \
        --value-col "sepal length (cm)" \
        --group-col species \
        --window 5 \
        --min-periods 1 \
        --out-csv iris_DA-3-29_roll_features.csv
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, List

import pandas as pd
import matplotlib.pyplot as plt

try:
    from DA_1_07 import load_iris_dataset, create_categorical_feature  # type: ignore
except Exception as e:
    raise ImportError(
        "Не удалось импортировать функции из DA_1_07.py. "
        "Убедитесь, что файл DA_1_07.py находится рядом и содержит функции "
        "`load_iris_dataset` и `create_categorical_feature`."
    ) from e


def compute_rolling_stats(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    window: int = 5,
    min_periods: int = 1,
    stats: Iterable[str] = ("mean", "min", "max", "std"),
) -> pd.DataFrame:
    """
    Добавляет rolling-агрегаты (mean, min, max, std) по каждому значению group_col.
    Возвращает копию df с новыми столбцами: {value_col}_roll_{stat}{window}.

    Параметры
    ---------
    df : pd.DataFrame
        Исходные данные, должны содержать столбцы value_col и group_col.
    value_col : str
        Числовой признак, для которого считаем статистики.
    group_col : str
        Категориальный столбец, по которому группируем.
    window : int
        Размер окна (число наблюдений) внутри каждой группы.
    min_periods : int
        Минимум наблюдений в окне для вычисления значения.
    stats : Iterable[str]
        Список статистик из {mean, min, max, std}.

    Исключения
    ----------
    ValueError: при некорректных параметрах.
    KeyError: если отсутствуют необходимые столбцы.
    """
    if not isinstance(window, int) or window <= 0:
        raise ValueError(f"window должно быть положительным целым числом, получено: {window}")
    if not isinstance(min_periods, int) or min_periods <= 0 or min_periods > window:
        raise ValueError(f"min_periods должно быть в диапазоне [1, window], получено: {min_periods}")
    if value_col not in df.columns:
        raise KeyError(f"Колонка '{value_col}' отсутствует в DataFrame")
    if group_col not in df.columns:
        raise KeyError(f"Колонка-группа '{group_col}' отсутствует в DataFrame")

    supported = {"mean", "min", "max", "std"}
    invalid = [s for s in stats if s not in supported]
    if invalid:
        raise ValueError(f"Неподдерживаемые статистики: {invalid}. Доступны: {sorted(supported)}")

    # Проверим, что признак числовой
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise ValueError(f"Колонка '{value_col}' не числовая и не подходит для rolling-агрегаций")

    df_out = df.copy()
    pieces: List[pd.DataFrame] = []

    for key, group in df_out.groupby(group_col, observed=False):
        # Сохраняем порядок как есть (в контексте задания индекс трактуем как время)
        g = group.sort_index()

        roll = g[value_col].rolling(window=window, min_periods=min_periods)

        # Вычисляем выбранные статистики
        if "mean" in stats:
            g[f"{value_col}_roll_mean{window}"] = roll.mean()
        if "min" in stats:
            g[f"{value_col}_roll_min{window}"] = roll.min()
        if "max" in stats:
            g[f"{value_col}_roll_max{window}"] = roll.max()
        if "std" in stats:
            g[f"{value_col}_roll_std{window}"] = roll.std()

        pieces.append(g)

    return pd.concat(pieces, axis=0).sort_index(kind="stable")


def plot_rolling_stats(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    window: int = 5,
    stats: Iterable[str] = ("mean", "min", "max", "std"),
    show: bool = True,
) -> None:
    """
    Рисует по одному графику для каждой статистики из `stats`.
    Линии — отдельные группы (значения group_col).
    """
    for stat in stats:
        col = f"{value_col}_roll_{stat}{window}"
        if col not in df.columns:
            print(f"[WARN] Пропускаю график для '{stat}': колонка '{col}' не найдена.", file=sys.stderr)
            continue

        plt.figure(figsize=(8, 3))
        for key, group in df.groupby(group_col, observed=False):
            plt.plot(group.index, group[col], label=str(key))
        plt.title(f"{value_col}: Rolling {stat.upper()} (window={window})")
        plt.xlabel("index (внутри группы)")
        plt.ylabel(stat)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
    if show:
        plt.show()


def save_dataframe_csv(df: pd.DataFrame, path: str) -> None:
    """Безопасное сохранение в CSV с обработкой ошибок файловой системы."""
    try:
        df.to_csv(path, index=False)
    except OSError as e:
        raise OSError(f"Не удалось сохранить CSV по пути '{path}': {e}") from e


def build_dataset_from_DA_1_07() -> pd.DataFrame:
    """Загружает iris и формирует DataFrame с колонкой 'species' через DA_1_07.py."""
    try:
        data, feature_names, target = load_iris_dataset()
        df = create_categorical_feature(data, feature_names, target)
    except Exception as e:
        raise RuntimeError(
            "Ошибка при подготовке датасета через DA_1_07.py. "
            "Проверьте реализованные там функции и совместимость."
        ) from e
    return df


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DA-3-29: Rolling-агрегаты (mean/min/max/std) по группам и графики."
    )
    parser.add_argument("--value-col", required=False, default="sepal length (cm)",
                        help="Числовая колонка для rolling (по умолчанию 'sepal length (cm)')")
    parser.add_argument("--group-col", required=False, default="species",
                        help="Категориальная колонка для группировки (по умолчанию 'species')")
    parser.add_argument("--window", type=int, default=5,
                        help="Размер окна (по умолчанию 5)")
    parser.add_argument("--min-periods", type=int, default=1,
                        help="Мин. наблюдений в окне (по умолчанию 1)")
    parser.add_argument("--out-csv", default="iris_DA-3-29_roll_features.csv",
                        help="Путь к выходному CSV (по умолчанию iris_DA-3-29_roll_features.csv)")
    parser.add_argument("--no-show", action="store_true",
                        help="Не показывать графики (полезно для headless-запуска)")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    # 1) Данные + категориальный признак (используем реализацию из DA_1_07)
    df = build_dataset_from_DA_1_07()

    # 2) Добавляем rolling-признаки
    try:
        df_roll = compute_rolling_stats(
            df=df,
            value_col=args.value_col,
            group_col=args.group_col,
            window=args.window,
            min_periods=args.min_periods,
            stats=("mean", "min", "max", "std"),
        )
    except (ValueError, KeyError) as e:
        print(f"[ERROR] Некорректные параметры или данные: {e}", file=sys.stderr)
        return 2

    # 3) Строим графики
    try:
        plot_rolling_stats(
            df_roll,
            value_col=args.value_col,
            group_col=args.group_col,
            window=args.window,
            stats=("mean", "min", "max", "std"),
            show=not args.no_show,
        )
    except Exception as e:
        print(f"[WARN] Не удалось построить графики: {e}", file=sys.stderr)

    # 4) Сохраняем в CSV
    try:
        save_dataframe_csv(df_roll, args.out_csv)
    except OSError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 3

    # Короткий вывод для контроля
    print(f"Сохранено: {args.out_csv}")
    with pd.option_context("display.max_columns", None):
        print(df_roll.head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())