# -*- coding: utf-8 -*-
import logging
from typing import Union, Optional, List, Dict

import pandas
from sklearn.base import BaseEstimator, RegressorMixin

from hts._t import Transform, Model
from hts.exceptions import InvalidArgumentException
from hts.functions import to_sum_mat
from hts.hierarchy import HierarchyTree
from hts.model.ar import AutoArimaModel, SarimaxModel
from hts.model.es import HoltWintersModel
from hts.model.p import FBProphetModel
from hts.revision import Methods

__author__ = """Carlo Mazzaferro"""
__email__ = 'carlo.mazzaferro@gmail.com'
__version__ = '0.2.0'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HTSRegressor(BaseEstimator, RegressorMixin):

    """
    Main regressor class for scikit-hts. Likely the only import you'll need for using
    this project. It takes a pandas dataframe, the nodes specifying the hierarchies, model kind, revision
    method, and a few other parameters. See Examples to get an idea of how to use it.


    Attributes
    ----------
    transform : Union[Callable, bool]
        Function transform to be applied to input and outputs. If True, it will use ``boxcox_transform`` from ``scipy``

    sum_mat : array_like
        The summing matrix, explained in depth in `Forecasting <https://otexts.com/fpp2/gts.html>`_

    nodes : Dict[str, List[str]]
        Nodes representing node, edges of the hierarchy. Keys are nodes, values are list of edges.


    df : pandas.DataFrame
        The dataframe containing the nodes and edges specified above

    revision_method : str
        One of the revisions methods specified in ...

    models : dict
        Dictionary that holds the trained models

    mse : dict
        Dictionary that holds the mse scores for the trained models

    residuals : dict
        Dictionary that holds the mse residual for the trained models

    forecasts : dict
        Dictionary that holds the forecasts for the trained models

    model_instance : TimeSeriesModel
        Reference to the class implementing the actual time series model

    Methods
    -------
    colorspace(c='rgb')
        Represent the photo in the given colorspace.
    gamma(n=1.0)
        Change the photo's gamma exposure.

    """
    def __init__(self,
                 df: pandas.DataFrame,
                 nodes: Dict[str, List[str]],
                 model: str = 'prophet',
                 periods: int = 1,
                 revision_method: str = 'OLS',
                 exogenous: Dict[str, List[str]] = None,
                 transform: Optional[Union[Transform, bool]] = None,
                 n_jobs: int = -1,
                 root: Union[str, HierarchyTree] = 'total',
                 **kwargs
                 ):
        """

        Parameters
        ----------
        df : pandas.DataFrame
        nodes : Dict[str, List[str]]

        model : str

        periods
        revision_method
        exogenous
        transform
        n_jobs
        root
        kwargs
        """

        self.model = model
        self.periods = periods
        self.method = revision_method
        self.n_jobs = n_jobs
        self.transform = transform
        self.sum_mat = None
        self.nodes = None
        self.df = None
        self.revision_method = Methods[self.method].value(**kwargs)
        self.models = dict()
        self.mse = dict()
        self.residuals = dict()
        self.forecasts = dict()
        self.model_instance = None
        self.__init_hts(nodes=nodes, df=df, root=root, exogenous=exogenous)
        self.n_forecasts = self.sum_mat.shape[0]
        self.model_args = kwargs

    def __init_hts(self, nodes, df, root, exogenous):
        self.nodes = HierarchyTree.from_nodes(nodes=nodes, df=df, exogenous=exogenous, root=root)
        self.sum_mat = to_sum_mat(self.nodes)
        self.get_model_instance()

    def get_model_instance(self):
        if self.model == Model.auto_arima.name:
            self.model_instance = AutoArimaModel
        elif self.model == Model.sarimax.name:
            self.model_instance = SarimaxModel
        elif self.model == Model.holt_winters.name:
            self.model_instance = HoltWintersModel
        elif self.model == Model.prophet.name:
            self.model_instance = FBProphetModel
        else:
            raise InvalidArgumentException(f'Model {self.model} not valid. Pick one of: {" ".join(Model.names())}')

    def fit_predict(self, **fit_args):
        for i, node in [self.nodes] + self.nodes.traversal_level():
            model = self.model_instance(node=node, transform=self.transform, **self.model_args)
            model.fit_predict(**fit_args)
            self.models[node.key] = model
            self.forecasts[node.key] = model.forecast
            self.mse[node.key] = model.mse
            self.residuals[node.key] = model.residual
        return self
