from scipy.special._ufuncs import inv_boxcox
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, RegressorMixin

from hts.core.functions import to_sum_mat
from hts.core.revision import Methods
from hts.transforms import FunctionTransformer
from hts.core.types import NAryTreeT


class HierarchicalProphet(BaseEstimator, RegressorMixin):

    """
    Parameters
    ----------
    periods
    nodes
    method
    freq
    transform
    include_history
    capacity
    capacity_future
    n_jobs
    kwargs
    """

    def __init__(self,
                 periods=1,
                 method='OLS',
                 freq='D',
                 transform=None,
                 include_history=True,
                 capacity=None,
                 capacity_future=None,
                 n_jobs=-1,
                 **kwargs):

        self.periods = periods
        self.freq = freq
        self.capacity = capacity
        self.capacity_future = capacity_future
        self.transform = transform
        if self.transform:
            if not isinstance(self.transform, dict):
                self.transformer = FunctionTransformer(func=boxcox,
                                                       inv_func=inv_boxcox)
            else:
                self.transformer = FunctionTransformer(transform)
        else:
            self.transformer = FunctionTransformer(func=lambda x: (x, None),
                                                   inv_func=lambda x: (x, None))
        self.include_history = include_history
        self.n_jobs = n_jobs
        self.method = method
        self.df = None
        self.sum_mat = None
        self.baseline = None
        self.model = None
        self.kwargs = kwargs

    @property
    def models(self):
        return self.model.models

    def _create_model(self, **kwargs):
        self.model = Methods[self.method].value(**kwargs)
        return self.model

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

    def fit(self, nodes: NAryTreeT):
        self.sum_mat = to_sum_mat(nodes)
        df = nodes.to_pandas().reset_index()
        self.df = self._transform(df)

        self._create_model(df=self.df,
                           sum_mat=self.sum_mat,
                           capacity=self.capacity,
                           capacity_future=self.capacity_future,
                           periods=self.periods,
                           freq=self.freq,
                           nodes=nodes,
                           transformer=self.transformer,
                           **self.kwargs)

        self.baseline = self.model.baseline_fit(self.df)
        if self.transform:
            for i, node in range(self.model.n_forecasts):
                self.baseline[node].yhat = self.transformer.inverse_transform(self.baseline[node].yhat)
                self.baseline[node].trend = self.transformer.inverse_transform(self.baseline[node].trend)
                for component in ["seasonal", "daily", "weekly", "yearly", "holidays"]:
                    if component in self.baseline[node].columns.tolist():
                        inv_transf = self.transformer.inverse_transform(getattr(self.baseline[node], component))
                        setattr(self.baseline[node], component, inv_transf)

    def predict_future(self):
        return self.model.predict()