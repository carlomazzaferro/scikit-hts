from enum import Enum

import numpy
import pandas
from fbprophet import Prophet

from hts import logger
from hts.core.functions import y_hat_matrix, optimal_combination, proportions, forecast_proportions


class RevisionMethod(object):

    def __init__(self,
                 df,
                 sum_mat,
                 capacity,
                 capacity_future,
                 periods=1,
                 freq='D',
                 nodes=None,
                 # changepoints=None,
                 transformer=None,
                 # n_changepoints=None,
                 **kwargs):

        self.df = df
        self.sum_mat = sum_mat
        self.n_forecasts = self.sum_mat.shape[0]
        self.nodes = nodes
        self.capacity = capacity
        self.capacity_future = capacity_future
        # self.changepoints = changepoints
        # self.n_changepoints = n_changepoints
        self.periods = periods
        self.freq = freq
        self.kwargs = kwargs
        self.transformer = transformer
        self.mse = dict()
        self.residuals = dict()
        self.forecasts = dict()

    def baseline_fit(self, df):
        for node in range(self.n_forecasts):
            node_to_forecast = pandas.concat([df.iloc[:, [0]], df.iloc[:, node + 1]], axis=1)
            logger.info(f'Producing forecast for node: {node_to_forecast.columns[1]}')
            capacity = self.capacity.iloc[:, node] if self.capacity else None
            capacity_future = self.capacity_future.iloc[:, node] if self.capacity_future else None
            # changepoints = self.changepoints[:, node]
            # n_changepoints = self.n_changepoints[node]

            # Put the forecasts into a dictionary of dataframes
            # with contextlib.redirect_stdout(open(os.devnull, "w")):
            # Prophet related stuff
            node_to_forecast = node_to_forecast.rename(columns={node_to_forecast.columns[0]: 'ds'})
            node_to_forecast = node_to_forecast.rename(columns={node_to_forecast.columns[1]: 'y'})
            if self.capacity_future is None:
                growth = 'linear'
            else:
                growth = 'logistic'

            m = Prophet(growth,
                        # changepoints,
                        # n_changepoints,
                        **self.kwargs)
            node_to_forecast['capacity'] = capacity
            m.fit(node_to_forecast)

            inlc_hist = self.kwargs['include_history'] if 'include_history' in self.kwargs else True
            future = m.make_future_dataframe(periods=self.periods, freq=self.freq,
                                             include_history=inlc_hist)
            if self.capacity_future is not None:
                future['capacity'] = capacity_future

            # Base Forecasts, Residuals, and MSE
            self.forecasts[node] = m.predict(future)

            self.residuals[node] = df.iloc[:, node + 1] - self.forecasts[node].yhat[:-self.periods].values
            self.mse[node] = numpy.mean(numpy.array(self.residuals[node]) ** 2)

            # If logistic use exponential function, so that values can be added correctly
            if self.capacity_future is not None:
                self.forecasts[node].yhat = numpy.exp(self.forecasts[node].yhat)
        return self.forecasts

    def predict(self):
        raise NotImplementedError

    def _reformat(self, transformed):
        for key in self.forecasts.keys():
            values = transformed[:, key]
            self.forecasts[key].yhat = values
            # If Logistic fit values with natural log function to revert back to format of input
            if self.capacity_future is not None:
                self.forecasts[key].yhat = numpy.log(self.forecasts[key].yhat)
        return self.forecasts

    def _new_mat(self, y_hat_mat):
        new_mat = numpy.empty([y_hat_mat.shape[0], self.sum_mat.shape[0]])
        for i in range(y_hat_mat.shape[0]):
            new_mat[i, :] = numpy.dot(self.sum_mat, numpy.transpose(y_hat_mat[i, :]))
        return new_mat


class CrossValidation(RevisionMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self):
        pass


class OLS(RevisionMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self):
        if not self.forecasts:
            raise ValueError('Baseline must be fit first')

        transformed = optimal_combination(forecasts=self.forecasts,
                                          sum_mat=self.sum_mat,
                                          method='OLS',
                                          mse=self.mse)
        return self._reformat(transformed)


class WLSS(RevisionMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self):
        if not self.forecasts:
            raise ValueError('Baseline must be fit first')

        transformed = optimal_combination(forecasts=self.forecasts,
                                          sum_mat=self.sum_mat,
                                          method=Methods.WLSS.name,
                                          mse=self.mse)
        return self._reformat(transformed)


class WLSV(RevisionMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self):
        if not self.forecasts:
            raise ValueError('Baseline must be fit first')

        transformed = optimal_combination(forecasts=self.forecasts,
                                          sum_mat=self.sum_mat,
                                          method=Methods.WLSV.name,
                                          mse=self.mse)
        return self._reformat(transformed)


class BU(RevisionMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _y_hat_matrix(self):
        n_cols = len(list(self.forecasts.keys())) + 1
        keys = range(n_cols - self.sum_mat.shape[1] - 1, n_cols - 1)
        return y_hat_matrix(self.forecasts, keys=keys)

    def predict(self):
        if not self.forecasts:
            raise ValueError('Baseline must be fit first')
        y_hat = self._y_hat_matrix()
        return self._new_mat(y_hat)


class AHP(RevisionMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self):
        if not self.forecasts:
            raise ValueError('Baseline must be fit first')
        if self.transformer:
            for column in range(len(self.df.columns.tolist()) - 1):
                self.df.iloc[:, column + 1] = self.transformer.inverse_transform(self.df.iloc[:, column + 1])
        y_hat = proportions(self.df, self.forecasts, self.sum_mat, method=Methods.AHP.name)
        return self._new_mat(y_hat)


class PHA(RevisionMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self):
        if not self.forecasts:
            raise ValueError('Baseline must be fit first')
        if self.transformer:
            for column in range(len(self.df.columns.tolist()) - 1):
                self.df.iloc[:, column + 1] = self.transformer.inverse_transform(self.df.iloc[:, column + 1])
        y_hat = proportions(self.df, self.forecasts, self.sum_mat, method=Methods.PHA.name)
        return self._new_mat(y_hat)


class FP(RevisionMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self):
        if not self.forecasts:
            raise ValueError('Baseline must be fit first')
        return forecast_proportions(self.forecasts, self.nodes)


class Methods(Enum):
    CV = CrossValidation
    OLS = OLS
    WLSS = WLSS
    WLSV = WLSV
    FP = FP
    PHA = PHA
    AHP = AHP
    BU = BU
