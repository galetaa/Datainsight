from sklearn.svm import OneClassSVM
from pandas import DataFrame


def one_class_svm(teacher: DataFrame, df: DataFrame, **kwargs):
    model = OneClassSVM(**kwargs)
    model.fit(teacher)
    predictions = model.predict(df)
    anomalies = df[predictions == -1]
    return anomalies
