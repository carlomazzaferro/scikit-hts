import logging
from typing import Optional, Union, Any

import pandas
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm

from hts.hierarchy.utils import make_iterable
from hts import model as hts_models
from hts._t import Transform, NodesT, ExogT, Model
from hts.core.exceptions import MissingRegressorException, InvalidArgumentException
from hts.core.result import HTSResult
from hts.functions import to_sum_mat
from hts.hierarchy import HierarchyTree
from hts.revision import RevisionMethod

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
                 **kwargs: Any
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
        self.model_instance = None
        self.exogenous = False
        self.revision_method = None
        self.hts_result = HTSResult()
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
            for node in make_iterable(self.nodes, prop=None):
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
        self._set_model_instance()
        self._init_revision()

    def _init_revision(self):
        self.revision_method = RevisionMethod(sum_mat=self.sum_mat, transformer=self.transform, name=self.method)

    def _set_model_instance(self):
        try:
            self.model_instance = hts_models.MODEL_MAPPING[self.model]
        except KeyError:
            raise InvalidArgumentException(f'Model {self.model} not valid. Pick one of: {" ".join(Model.names())}')

    def fit(self,
            df: pandas.DataFrame,
            nodes: NodesT,
            exogenous: Optional[ExogT] = None,
            root: str = 'total',
            **fit_args: Any) -> 'HTSRegressor':

        """
        Fit hierarchical model to dataframe containing hierarchical data as specified in the ``nodes`` parameter
        Exogenous can also be passed as a dict of (string, list), where string is the specific node key and the list
        contains the names of the columns to be used as exogenous variables for that node.

        Parameters
        ----------
        df : pandas.DataFrame
            A Dataframe of time series with a DateTimeIndex. Each column represents a node in the hierarchy
        nodes : Dict[str, List[str]]
            The hierarchy defined as a dict of (string, list), as specified in
             :py:func:`HierarchyTree.from_nodes <hts.hierarchy.HierarchyTree.from_nodes>`
        exogenous : Dict[str, List[str]] or None
            Node key mapping to columns that contain the exogenous variable for that node
        root : str
            The name of the root node
        fit_args: Any
            Any arguments to be passed to the underlying forecasting model's fit function. You will have to

        Returns
        -------
        HTSRegressor
            The fitted HTSRegressor instance
        """

        self.__init_hts(nodes=nodes, df=df, root=root, exogenous=exogenous)

        iterable = tqdm(make_iterable(self.nodes, prop=None))

        for node in iterable:
            self._fit_step(node, iterable, **fit_args)
        return self

    def _fit_step(self, node: HierarchyTree, iterable: tqdm, **fit_args):
        model_instance = self.model_instance(node=node, transform=self.transform, **self.model_args)
        iterable.set_description(f'Fitting base model for node : {node.key}')
        model_instance = model_instance.fit(**fit_args)
        self.hts_result.models = (node.key, model_instance)

    def predict(self,
                exogenous_df: pandas.DataFrame = None,
                steps_ahead: int = None,
                **predict_kwargs) -> pandas.DataFrame:
        """

        Parameters
        ----------
        exogenous_df
        steps_ahead
        predict_kwargs

        Returns
        -------
        Revised Forescasts, as a pandas.DataFrame in the same format as the one passed for fitting, extended by `steps_ahead`
        time steps`
        """

        steps_ahead = self.__init_predict_step(exogenous_df, steps_ahead)
        iterable = tqdm(make_iterable(self.nodes, prop=None))

        for node in iterable:
            self._predict_step(node, iterable=iterable, steps_ahead=steps_ahead, **predict_kwargs)
        return self._revise(steps_ahead=steps_ahead)

    def _predict_step(self, node: HierarchyTree, iterable: tqdm, steps_ahead: int, **predict_kwargs):
        model_instance = self.hts_result.models[node.key]
        iterable.set_description(f'Generating base prediction for node: {node.key}')
        model_instance = model_instance.predict(node=node, steps_ahead=steps_ahead, **predict_kwargs)
        self.hts_result.forecasts = (node.key, model_instance.forecast)
        self.hts_result.errors = (node.key, model_instance.mse)
        self.hts_result.residuals = (node.key, model_instance.residual)

    def _revise(self, steps_ahead=1):
        logger.info(f'Reconciling forecasts using {self.revision_method}')
        revised = self.revision_method.revise(
            forecasts=self.hts_result.forecasts,
            mse=self.hts_result.errors,
            nodes=self.nodes
        )

        revised_columns = list(make_iterable(self.nodes))
        revised_index = self._get_predict_index(steps_ahead=steps_ahead)
        return pandas.DataFrame(revised,
                                index=revised_index,
                                columns=revised_columns)

    def _get_predict_index(self, steps_ahead=1):
        freq = pandas.infer_freq(self.nodes.item.index)
        future = pandas.date_range(freq=freq,
                                   start=self.nodes.item.index.max() + pandas.Timedelta(1, freq),
                                   end=self.nodes.item.index.max() + pandas.Timedelta(steps_ahead, freq)
                                   )

        return self.nodes.item.index.append(future)
