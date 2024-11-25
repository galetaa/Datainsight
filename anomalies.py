from pandas import DataFrame
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from statsmodels.tsa.seasonal import seasonal_decompose


def one_class_svm(teacher: DataFrame, df: DataFrame, **kwargs):
    model = OneClassSVM(**kwargs)
    model.fit(teacher)
    predictions = model.predict(df)
    anomalies = df[predictions == -1]
    return anomalies


class AnomaliesDetector:
    def __init__(self, data: DataFrame | None = None):
        self.data = data

    def load_data(self, data: DataFrame):
        self.data = data

    def detect_anomalies_3sigma(self, column):
        df = self.data

        mean = df[column].mean()
        std = df[column].std()
        anomalies = df[(df[column] < mean - 3 * std) | (df[column] > mean + 3 * std)]
        return anomalies

    def detect_anomalies_iqr(self, column):
        df = self.data

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        anomalies = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
        return anomalies

    def detect_anomalies_zscore(self, column, threshold=3):
        df = self.data

        df['z_score'] = zscore(df[column])
        anomalies = df[df['z_score'].abs() > threshold]
        return anomalies.drop(columns=['z_score'])

    def detect_anomalies_isolation_forest(self, columns):
        df = self.data

        model = IsolationForest(random_state=42)
        df['anomaly'] = model.fit_predict(df[columns])
        anomalies = df[df['anomaly'] == -1]
        return anomalies.drop(columns=['anomaly'])

    def detect_anomalies_svm(self, columns):
        df = self.data

        model = OneClassSVM(kernel='rbf', gamma='auto')
        df['anomaly'] = model.fit_predict(df[columns])
        anomalies = df[df['anomaly'] == -1]
        return anomalies.drop(columns=['anomaly'])

    def detect_anomalies_dbscan(self, columns, eps=0.5, min_samples=5):
        df = self.data

        model = DBSCAN(eps=eps, min_samples=min_samples)
        df['anomaly'] = model.fit_predict(df[columns])
        anomalies = df[df['anomaly'] == -1]
        return anomalies.drop(columns=['anomaly'])

    def detect_anomalies_knn(self, columns, n_neighbors=5, threshold=1.5):
        df = self.data

        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(df[columns])
        distances, _ = knn.kneighbors(df[columns])
        avg_distances = distances.mean(axis=1)
        df['distance'] = avg_distances
        anomalies = df[df['distance'] > threshold]
        return anomalies.drop(columns=['distance'])

    def detect_anomalies_stl(self, column, period, threshold=1.5):
        df = self.data

        decomposition = seasonal_decompose(df[column], period=period)
        residual = decomposition.resid
        df['residual'] = residual
        anomalies = df[residual.abs() > threshold]
        return anomalies.drop(columns=['residual'])

    def detect_anomalies(self, method, column=None, columns=None, **kwargs):
        if method == '3sigma':
            return self.detect_anomalies_3sigma(column)
        elif method == 'iqr':
            return self.detect_anomalies_iqr(column)
        elif method == 'zscore':
            return self.detect_anomalies_zscore(column, **kwargs)
        elif method == 'isolation_forest':
            return self.detect_anomalies_isolation_forest(columns)
        elif method == 'svm':
            return self.detect_anomalies_svm(columns)
        elif method == 'dbscan':
            return self.detect_anomalies_dbscan(columns, **kwargs)
        elif method == 'knn':
            return self.detect_anomalies_knn(columns, **kwargs)
        elif method == 'stl':
            return self.detect_anomalies_stl(column, **kwargs)
        else:
            raise ValueError(f"Метод '{method}' не поддерживается.")
