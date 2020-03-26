# -*- coding: utf-8 -*-
import logging
from typing import Union, Optional, List, Dict

import pandas
from tqdm import tqdm
from sklearn.base import BaseEstimator, RegressorMixin

from hts._t import Transform, Model, NodesT, ExogT
from hts.exceptions import InvalidArgumentException, MissingRegressorException
from hts.functions import to_sum_mat
from hts.hierarchy import HierarchyTree
from hts.revision import RevisionMethod
from hts.model import MODEL_MAPPING


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
        self.revision_method = None
        self.revised_forecasts = None
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
        self.df = df
        self.sum_mat = to_sum_mat(self.nodes)
        self.n_forecasts = self.sum_mat.shape[0]
        self._set_model_instance()
        self._init_revision()

    def _init_revision(self):
        self.revision_method = RevisionMethod(sum_mat=self.sum_mat, transformer=self.transform, name=self.method)

    def _set_model_instance(self):
        try:
            self.model_instance = MODEL_MAPPING[self.model]
        except KeyError:
            raise InvalidArgumentException(f'Model {self.model} not valid. Pick one of: {" ".join(Model.names())}')

    def fit(self,
            df: pandas.DataFrame,
            nodes: NodesT,
            exogenous: Optional[ExogT] = None,
            root: Union[str, HierarchyTree] = 'total',
            **fit_args) -> 'HTSRegressor':
        self.__init_hts(nodes=nodes, df=df, root=root, exogenous=exogenous)

        to_fit = tqdm([self.nodes] + self.nodes.traversal_level())

        for node in to_fit:
            model_instance = self.model_instance(node=node, transform=self.transform, **self.model_args)
            to_fit.set_description(f'Fitting base model for node: {node.key}')
            model_instance = model_instance.fit(**fit_args)
            self.models[node.key] = model_instance
        return self

    def predict(self,
                exogenous_df: pandas.DataFrame = None,
                steps_ahead: int = None,
                **predict_kwargs) -> pandas.DataFrame:

        steps_ahead = self.__init_predict_step(exogenous_df, steps_ahead)
        to_predict = tqdm([self.nodes] + self.nodes.traversal_level())

        for node in to_predict:
            model_instance = self.models[node.key]
            to_predict.set_description(f'Generating base prediction for node: {node.key}')
            model_instance = model_instance.predict(node=node, steps_ahead=steps_ahead, **predict_kwargs)
            self.models[node.key] = model_instance
            self.forecasts[node.key] = model_instance.forecast
            self.mse[node.key] = model_instance.mse
            self.residuals[node.key] = model_instance.residual
        return self._revise(steps_ahead=steps_ahead)

    def _revise(self, steps_ahead=1):
        revised = self.revision_method.revise(
            forecasts=self.forecasts, mse=self.mse, df=self.df, nodes=self.nodes
        )
        revised_columns = ['total'] + [k.key for k in self.nodes.traversal_level()]
        revised_index = self._get_predict_index(steps_ahead=steps_ahead)
        return pandas.DataFrame(revised,
                                index=revised_index,
                                columns=revised_columns)

    def _get_predict_index(self, steps_ahead=1):
        freq = pandas.infer_freq(self.df.index)
        future = pandas.date_range(freq=freq,
                                   start=self.df.index.max() + pandas.Timedelta(1, freq),
                                   end=self.df.index.max() + pandas.Timedelta(steps_ahead, freq)
                                   )

        return self.df.index.append(future)

