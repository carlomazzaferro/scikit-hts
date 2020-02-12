from enum import Enum

import numpy

from hts.functions import y_hat_matrix, optimal_combination, proportions, forecast_proportions


class RevisionMethod(object):

    def __init__(self,
                 df,
                 forecasts,
                 mse,
                 sum_mat,
                 transformer,
                 ):
        self.df = df
        self.forecasts = forecasts
        self.transformer = transformer
        self.mse = mse
        self.sum_mat = sum_mat

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

        return optimal_combination(forecasts=self.forecasts,
                                   sum_mat=self.sum_mat,
                                   method=Methods.OLS.name,
                                   mse=self.mse)


class WLSS(RevisionMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self):
        if not self.forecasts:
            raise ValueError('Baseline must be fit first')

        return optimal_combination(forecasts=self.forecasts,
                                   sum_mat=self.sum_mat,
                                   method=Methods.WLSS.name,
                                   mse=self.mse)


class WLSV(RevisionMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self):
        if not self.forecasts:
            raise ValueError('Baseline must be fit first')

        return optimal_combination(forecasts=self.forecasts,
                                   sum_mat=self.sum_mat,
                                   method=Methods.WLSV.name,
                                   mse=self.mse)


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
