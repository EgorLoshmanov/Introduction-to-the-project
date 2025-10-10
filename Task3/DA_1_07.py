import pandas as pd
from sklearn.datasets import load_iris

def load_iris_dataset():
    """
    Загружает датасет Iris из scikit-learn.

    Возвращает:
        tuple: Данные Iris (numpy.ndarray) и имена признаков (list).
    """
    try:
        iris = load_iris()
        return iris.data, iris.feature_names, iris.target
    except Exception as e:
        raise Exception(f"Не удалось загрузить датасет Iris: {str(e)}")

def create_categorical_feature(data, feature_names, target):
    """
    Создаёт DataFrame с категориальным признаком 'species' на основе целевой переменной.

    Аргументы:
        data (numpy.ndarray): Данные признаков Iris.
        target (numpy.ndarray): Целевая переменная (виды).

    Возвращает:
        pandas.DataFrame: DataFrame с добавленным столбцом 'species'.
    """
    df = pd.DataFrame(data=data, columns=feature_names)
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    try:
        df['species'] = pd.Categorical(
            [species_map[target_val] for target_val in target],
            categories=['setosa', 'versicolor', 'virginica'],
            ordered=False
        )
    except KeyError as e:
        raise KeyError(f"Ошибка в создании категориального признака: {str(e)}")
    return df

def aggregate_by_species(df):
    """
    Группирует данные по видам и вычисляет агрегации для признаков.

    Аргументы:
        df (pandas.DataFrame): DataFrame с данными Iris и столбцом 'species'.

    Возвращает:
        pandas.DataFrame: Результаты агрегации (mean, sum, count, min, max).
    """
    if 'species' not in df.columns:
        raise ValueError("Столбец 'species' отсутствует в DataFrame")
    result = df.groupby('species', observed=False).agg({
        'sepal length (cm)': ['mean', 'sum', 'count', 'min', 'max'],
        'sepal width (cm)': ['mean', 'sum', 'count', 'min', 'max'],
        'petal length (cm)': ['mean', 'sum', 'count', 'min', 'max'],
        'petal width (cm)': ['mean', 'sum', 'count', 'min', 'max']
    })
    return result

def save_and_display_result(result):
    """
    Сохраняет результаты агрегации в CSV-файл и выводит их в консоль.

    Аргументы:
        result (pandas.DataFrame): DataFrame с результатами агрегации.
    """
    try:
        result.to_csv('iris_analysis_result.csv')
        print(result)
    except IOError as e:
        raise IOError(f"Ошибка при сохранении результата в CSV: {str(e)}")

if __name__ == "__main__":
    try:
        # 1. Загрузка данных
        data, feature_names, target = load_iris_dataset()
        # 2. Создание категориального признака
        df = create_categorical_feature(data, feature_names, target)
        # 3. Группировка и агрегация
        result = aggregate_by_species(df)
        # 4. Вывод и сохранение результата
        save_and_display_result(result)
    except Exception as e:
        print(f"Ошибка: {str(e)}")