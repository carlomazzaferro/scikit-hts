
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.base import BaseEstimator, RegressorMixin

from hts.algos.foreacast.method import Methods
from hts.algos.foreacast.helpers import summing_mat
from hts.algos.transforms import FunctionTransformer


class HierarchicalProphet(BaseEstimator, RegressorMixin):

    def __init__(self,
                 periods=1,
                 nodes=None,
                 method='OLS',
                 freq='D',
                 transform=None,
                 include_history=True,
                 capacity=None,
                 capacity_future=None,
                 n_jobs=-1,
                 **kwargs):

        if nodes is None:
            nodes = [[2]]
        self.nodes = nodes
        self.periods = periods
        self.freq = freq
        self.capacity = capacity
        self.capacity_future = capacity_future
        self.transformer = None
        self.transform = transform
        if self.transform:
            if not isinstance(self.transform, dict):
                self.transformer = FunctionTransformer(func=boxcox,
                                                       inv_func=inv_boxcox)
            else:
                self.transformer = FunctionTransformer(transform)
        self.include_history = include_history
        self.n_jobs = n_jobs
        self.method = method
        self.sum_mat = summing_mat(nodes)
        self.df = None
        self.baseline = None
        self.model = None
        self.kwargs = kwargs

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

    def fit(self, df):
        self.df = self._transform(df)
        self._create_model(df=self.df,
                           sum_mat=self.sum_mat,
                           capacity=self.capacity,
                           capacity_future=self.capacity_future,
                           periods=self.periods,
                           freq=self.freq,
                           nodes=self.nodes,
                           transformer=self.transformer,
                           **self.kwargs)

        self.baseline = self.model.baseline_fit(self.df)
        if self.transform:
            for node in range(self.model.n_forecasts):
                self.baseline[node].yhat = self.transformer.inverse_transform(self.baseline[node].yhat)
                self.baseline[node].trend = self.transformer.inverse_transform(self.baseline[node].trend)
                for component in ["seasonal", "daily", "weekly", "yearly", "holidays"]:
                    if component in self.baseline[node].columns.tolist():
                        inv_transf = self.transformer.inverse_transform(getattr(self.baseline[node], component))
                        setattr(self.baseline[node], component, inv_transf)

    def predict_future(self):
        return self.model.predict()
