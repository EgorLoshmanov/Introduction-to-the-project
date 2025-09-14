import pandas as pd
from sklearn.datasets import load_iris

def get_iris_slice(row_count: int = 10, cols: tuple = (0, 1)) -> pd.DataFrame:
    """
    Возвращает срез данных из датасета Iris как DataFrame.

    :param row_count: количество первых строк (по умолчанию: 10)
    :param cols: индексы столбцов (по умолчанию: (0,1))
    :return: pandas.DataFrame
    :raises ValueError: если параметры некорректны
    """
    try:
        iris = load_iris(as_frame=True)
        df = iris.data
        n_rows, n_cols = df.shape

        # --- Проверка аргументов ---
        if not isinstance(row_count, int) or row_count <= 0:
            raise ValueError("row_count должен быть положительным целым числом")
        if row_count > n_rows:
            raise ValueError(f"row_count не может превышать количество строк ({n_rows})")

        if not isinstance(cols, (list, tuple)) or not all(isinstance(c, int) for c in cols):
            raise ValueError("cols должен быть списком или кортежем целых чисел")
        if any(c < 0 or c >= n_cols for c in cols):
            raise ValueError(f"Каждый индекс из cols должен быть в диапазоне 0..{n_cols-1}")

        # --- Основная логика ---
        return df.iloc[:row_count, list(cols)]

    except Exception as e:
        # Перехватываем любые неожиданные ошибки и оборачиваем в RuntimeError
        raise RuntimeError(f"Не удалось получить срез данных Iris: {e}") from e


# Пример использования
if __name__ == "__main__":
    try:
        subset = get_iris_slice(row_count=10, cols=(0, 1))
        print(subset)
    except Exception as err:
        print(f"Ошибка: {err}")
