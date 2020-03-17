# -*- coding: utf-8 -*-
import logging
from typing import Union, Optional, List, Dict

import pandas
from sklearn.base import BaseEstimator, RegressorMixin

from hts._t import Transform, Model, NodesT, ExogT
from hts.exceptions import InvalidArgumentException, MissingRegressorException
from hts.functions import to_sum_mat
from hts.hierarchy import HierarchyTree
from hts.model.ar import AutoArimaModel, SarimaxModel
from hts.model.es import HoltWintersModel
from hts.model.p import FBProphetModel
from hts.revision import Methods

__author__ = """Carlo Mazzaferro"""
__email__ = 'carlo.mazzaferro@gmail.com'
__version__ = '0.2.1'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HTSRegressor(BaseEstimator, RegressorMixin):
    """
    Main regressor class for scikit-hts. Likely the only import you'll need for using
    this project. It takes a pandas dataframe, the nodes specifying the hierarchies, model kind, revision
    method, and a few other parameters. See Examples to get an idea of how to use it.


    Attributes
    ----------
    transform : Union[NamedTuple[str, Callable], bool]
        Function transform to be applied to input and outputs. If True, it will use ``scipy.stats.boxcox``
         and ``scipy.special._ufuncs.inv_boxcox`` on input and output data

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
    fit(self, df: pandas.DataFrame, nodes: Dict, exogenous: List = None, root: str ='total', **fit_args)
        Fits underlying models to the data

    predict(self, exogenous: pandas.DataFrame = None, steps_ahead: int = 10)
        Predicts the n-step ahead forecast with exogenous variables. Exogenous variables are required if models were
        fit using them
    """

    def __init__(self,
                 model: str = 'prophet',
                 periods: int = 1,
                 revision_method: str = 'OLS',
                 transform: Optional[Union[Transform, bool]] = None,
                 n_jobs: int = -1,
                 **kwargs
                 ):
        """

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe containing the nodes and edges specified in nodes

        model : str
            One of the models supported by ``hts``. These can be found

        periods : int
        revision_method : str
        transform : Boolean or NamedTuple
            If True, ``scipy.stats.boxcox`` and ``scipy.special._ufuncs.inv_boxcox`` will be applied prior and after
            fitting.
            If False, no transform is applied.
            If you desired to use custom functions, use a NamedTuple like: ``{'func': Callable, 'inv_func': Callable}``
        n_jobs : int
            Number of parallel jobs to run the forecasting on
        root : str
            The name of the root node. Must be one of the dataframe's columns names
        kwargs
            Keyword arguments to be passed to the underlying model to be instantiated
        """

        self.model = model
        self.periods = periods
        self.method = revision_method
        self.n_jobs = n_jobs
        self.transform = transform
        self.sum_mat = None
        self.nodes = None
        self.df = None
        self.n_forecasts = None
        self.model_instance = None
        self.exogenous = False
        self.revision_method = Methods[self.method].value(**kwargs)
        self.models = dict()
        self.mse = dict()
        self.residuals = dict()
        self.forecasts = dict()
        self.model_args = kwargs

    def __init_predict_step(self, exogenous_df: pandas.DataFrame, steps_ahead: int):
        if self.exogenous and not exogenous_df:
            raise MissingRegressorException(f'Exogenous variables were provided at fit step, hence are required at '
                                            f'predict step. Please pass the \'exogenous_df\' variable to predict '
                                            f'function')
        if not exogenous_df and not steps_ahead:
            logger.info(f'No arguments passed for \'steps_ahead\', defaulting to predicting 1-step-ahead')
            steps_ahead = 1
        elif exogenous_df:
            steps_ahead = len(exogenous_df)
            for node in [self.nodes] + self.nodes.traversal_level():
                exog_cols = node.exogenous
                try:
                    node.item = exogenous_df[exog_cols]
                except KeyError:
                    raise MissingRegressorException(f'Node {node.key} has as exogenous variables {node.exogenous} but '
                                                    f'these columns were not found in \'exogenous_df\'')
        return steps_ahead

    def __init_hts(self, nodes, df, root, exogenous=None):
        self.exogenous = exogenous
        self.nodes = HierarchyTree.from_nodes(nodes=nodes, df=df, exogenous=exogenous, root=root)
        self.sum_mat = to_sum_mat(self.nodes)
        self.n_forecasts = self.sum_mat.shape[0]
        self._get_model_instance()

    def _get_model_instance(self):
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

    def fit(self,
            df: pandas.DataFrame,
            nodes: NodesT,
            exogenous: Optional[ExogT] = None,
            root: Union[str, HierarchyTree] = 'total',
            **fit_args) -> 'HTSRegressor':
        self.__init_hts(nodes=nodes, df=df, root=root, exogenous=exogenous)
        for node in [self.nodes] + self.nodes.traversal_level():
            model = self.model_instance(node=node, transform=self.transform, **self.model_args)
            model = model.fit(**fit_args)
            self.models[node.key] = model
        return self

    def predict(self,
                exogenous_df: pandas.DataFrame = None,
                steps_ahead: int = None,
                **predict_kwargs) -> 'HTSRegressor':
        steps_ahead = self.__init_predict_step(exogenous_df, steps_ahead)
        for node in [self.nodes] + self.nodes.traversal_level():
            model = self.models[node.key]
            model = model.predict(node=node, steps_ahead=steps_ahead, **predict_kwargs)
            self.models[node.key] = model
            self.forecasts[node.key] = model.forecast
            self.mse[node.key] = model.mse
            self.residuals[node.key] = model.residual
        return self
