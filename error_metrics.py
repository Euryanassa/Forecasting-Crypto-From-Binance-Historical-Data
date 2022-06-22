import numpy as np

class error_metrics:
    def mape(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def mse(y_true, y_pred):
        return np.square(np.subtract(y_true, y_pred)).mean()

    def mae(y_true, y_pred):
        return np.sum(np.absolute((y_true - y_pred)))

    def rmse(y_true, y_pred):
        rms = np.sqrt(error_metrics.mse(y_true, y_pred))
        return rms