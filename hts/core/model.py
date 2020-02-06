from typing import Callable, Union, Optional
from typing_extensions import Literal
import pandas
from scipy.stats import boxcox
from scipy.special._ufuncs import inv_boxcox
from sklearn.base import BaseEstimator, RegressorMixin

from hts.core.factory import TimeSeriesModel
from hts.core.functions import to_sum_mat
from hts.core.revision import Methods
from hts._t import NAryTreeT, Transform, Models
from hts.transforms import FunctionTransformer


class HTS(BaseEstimator, RegressorMixin):

    def __init__(self,
                 nodes: NAryTreeT,
                 model: str = 'prophet',
                 periods: int = 1,
                 revision_method: str = 'OLS',
                 freq: str = 'D',
                 transform: Optional[Union[Transform, bool]] = None,
                 n_jobs: int = -1,
                 **kwargs
                 ):
        self.model = model
        self.periods = periods
        self.method = revision_method
        self.n_jobs = -1
        if transform is not None:
            if transform is True:
                self.transformer = FunctionTransformer(func=boxcox,
                                                       inv_func=inv_boxcox)
            else:
                self.transformer = FunctionTransformer(func=transform.func,
                                                       inv_func=transform.inv_func)
        else:
            self.transformer = FunctionTransformer(func=lambda x: (x, None),
                                                   inv_func=lambda x: (x, None))

        self.sum_mat = None
        self.nodes = nodes
        self.df = None
        self.revision_method = Methods[self.method].value(**kwargs)
        self.models = dict()
        self.mse = dict()
        self.residuals = dict()
        self.forecasts = dict()
        self.__init_hts()

    def _transform(self, df):
        time = df.columns[0]
        other = df.columns[1:]
        transf = self.transformer.transform(df[other])
        transf[time] = df[time]
        return df

    def _inv_transform(self, result):
        time = result.columns[0]
        other = result.columns[1:]
        transf = self.transformer.inverse_transform(result[other])
        transf[time] = result[time]
        return result

    def __init_hts(self):
        self.sum_mat = to_sum_mat(self.nodes)
        df = nodes.to_pandas().reset_index()
        self.df = self._transform(df)

    def fit(self):
        for i, node in [self.nodes] + self.nodes.traversal_level():
            model = TimeSeriesModel(kind=self.model, data=node)



class Forecast:

    def __init__(self, df: pandas.DataFrame):
        self.df = df

    @property
    def yhat(self):
        return self.df.yhat
