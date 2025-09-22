from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray

def make_cluster_with_centers(number_of_samples: int, number_of_centers: int, random_state: int = 42) -> tuple[NDArray, NDArray, NDArray]:
    """
    Генерирует 2D-данные и считает центры кластеров

    Возвращает:
        X        - координаты точек, shape (n_samples, 2)
        y        - метки кластеров, shape (n_samples,)
        centers  - центры кластеров, shape (max_label+1, 2)
                   (если каких-то меток нет, соответствующие строки будут NaN)
    """
    X, y = make_blobs(n_samples=number_of_samples,centers=number_of_centers,n_features=2, random_state=random_state)

    n_features = X.shape[1]
    max_label = int(y.max())
    centers = np.full((max_label + 1, n_features), np.nan, dtype=float)
    for k in np.unique(y):
        centers[k] = X[y == k].mean(axis=0)

    return X, y, centers

def run_pipeline( number_of_samples: int, number_of_centers: int, random_state: int = 42) -> tuple[pd.DataFrame, NDArray, NDArray]:
    """
    Основная логика:
      1) Генерирует данные и центры
      2) Считает расстояния до центров
      3) Возвращает (df, centers, distances) 
    """
    # --- базовая валидация входов ---
    if not isinstance(number_of_samples, int) or not isinstance(number_of_centers, int):
        raise TypeError("number_of_samples и number_of_centers должны быть int")
    if number_of_samples < 1 or number_of_centers < 1:
        raise ValueError("Оба параметра должны быть >= 1.")
    if number_of_samples < number_of_centers:
        raise ValueError("n_samples должно быть ≥ n_centers (иначе возможны пустые кластеры).")

    # --- генерация и центры ---
    X, y, centers = make_cluster_with_centers(number_of_samples, number_of_centers, random_state)

    # --- расстояния до собственных центров ---
    distances = np.linalg.norm(X - centers[y], axis=1)

    # --- таблица признаков ---
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["cluster"] = y
    df["dist_to_center"] = distances

    # быстрая сводка
    print(f"shape: {df.shape}")
    print(df.head(10))

    return df, centers, distances

def main() -> None:
    # пример вызова 
    df, centers, distances = run_pipeline(number_of_samples=500,number_of_centers=5, random_state=42)
  
if __name__ == "__main__":
    main()