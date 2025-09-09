import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.data

print(df.iloc[[x for x in range(10)], [0, 1]])