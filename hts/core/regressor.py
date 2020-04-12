import logging
import tempfile
from typing import Optional, Union, Any, Dict, List

import pandas
from sklearn.base import BaseEstimator, RegressorMixin

from hts import model as hts_models, defaults
from hts._t import Transform, NodesT, ExogT, Model
from hts.core.exceptions import MissingRegressorException, InvalidArgumentException
from hts.core.result import HTSResult
from hts.core.utils import _do_fit, _do_predict, _model_mapping_to_iterable
from hts.functions import to_sum_mat
from hts.hierarchy import HierarchyTree
from hts.hierarchy.utils import make_iterable
from hts.model.base import TimeSeriesModel
from hts.revision import RevisionMethod
from hts.utilities.distribution import DistributorBaseClass

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
                 model: str = defaults.MODEL,
                 revision_method: str = defaults.REVISION,
                 transform: Optional[Union[Transform, bool]] = None,
                 n_jobs: int = defaults.N_PROCESSES,
                 low_memory: bool = defaults.LOW_MEMORY,
                 **kwargs: Any
                 ):
        """
        Parameters
        ----------
        model : str
            One of the models supported by ``hts``. These can be found
        revision_method : str
            The revision method to be used
        transform : Boolean or NamedTuple
            If True, ``scipy.stats.boxcox`` and ``scipy.special._ufuncs.inv_boxcox`` will be applied prior and after
            fitting.
            If False, no transform is applied.
            If you desired to use custom functions, use a NamedTuple like: ``{'func': Callable, 'inv_func': Callable}``
        n_jobs : int
            Number of parallel jobs to run the forecasting on
        low_memory : Bool
            If True, models will be fit, serialized, and released from memory. Usually a good idea if
            you are dealing with a large amount of nodes
        kwargs
            Keyword arguments to be passed to the underlying model to be instantiated
        """

        self.model = model
        self.method = revision_method
        self.n_jobs = n_jobs
        self.low_memory = low_memory
        if self.low_memory:
            self.tmp_dir = tempfile.mkdtemp(prefix='hts_')
        else:
            self.tmp_dir = None
        self.transform = transform

        self.sum_mat = None
        self.nodes = None
        self.model_instance = None
        self.exogenous = False
        self.revision_method = None
        self.hts_result = HTSResult()
        self.model_args = kwargs

    def __init_hts(self,
                   nodes: Optional[NodesT] = None,
                   df: Optional[pandas.DataFrame] = None,
                   tree: Optional[HierarchyTree] = None,
                   root: str = 'root',
                   exogenous: Optional[List[str]] = None):

        if not nodes and not df:
            if not tree:
                raise InvalidArgumentException('Either nodes and df must be passed, or a pre-built hierarchy tree')
            else:
                self.nodes = tree
        else:
            self.nodes = HierarchyTree.from_nodes(nodes=nodes, df=df, exogenous=exogenous, root=root)
        self.exogenous = exogenous
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
            df: Optional[pandas.DataFrame] = None,
            nodes: Optional[NodesT] = None,
            tree: Optional[HierarchyTree] = None,
            exogenous: Optional[ExogT] = None,
            root: str = 'total',
            distributor: Optional[DistributorBaseClass] = None,
            disable_progressbar=defaults.DISABLE_PROGRESSBAR,
            show_warnings=defaults.SHOW_WARNINGS,
            **fit_kwargs: Any) -> 'HTSRegressor':

        """
        Fit hierarchical model to dataframe containing hierarchical data as specified in the ``nodes`` parameter
        Exogenous can also be passed as a dict of (string, list), where string is the specific node key and the list
        contains the names of the columns to be used as exogenous variables for that node.

        Alternatively, a pre-built HierarchyTree can be passed without specifying the node and df. See more at
        :class:`hts.hierarchy.HierarchyTree`

        Parameters
        ----------
        df : pandas.DataFrame
            A Dataframe of time series with a DateTimeIndex. Each column represents a node in the hierarchy. Ignored if
            tree argument is passed
        nodes : Dict[str, List[str]]
            The hierarchy defined as a dict of (string, list), as specified in
             :py:func:`HierarchyTree.from_nodes <hts.hierarchy.HierarchyTree.from_nodes>`
        tree : HierarchyTree
            A pre-built HierarchyTree. Ignored if df and nodes are passed, as the tree will be built from thise
        distributor : Optional[DistributorBaseClass]
             A distributor, for parallel/distributed processing
        exogenous : Dict[str, List[str]] or None
            Node key mapping to columns that contain the exogenous variable for that node
        root : str
            The name of the root node
        disable_progressbar : Bool
            Disable or enable progressbar
        show_warnings : Bool
            Disable warnings
        fit_kwargs : Any
            Any arguments to be passed to the underlying forecasting model's fit function

        Returns
        -------
        HTSRegressor
            The fitted HTSRegressor instance
        """

        self.__init_hts(nodes=nodes, df=df, tree=tree, root=root, exogenous=exogenous)

        nodes = make_iterable(self.nodes, prop=None)

        fit_function_kwargs = {'fit_kwargs': fit_kwargs,
                               'low_memory': self.low_memory,
                               'tmp_dir': self.tmp_dir,
                               'model_instance': self.model_instance,
                               'model_args': self.model_args,
                               'transform': self.transform
                               }

        fitted_models = _do_fit(
            nodes=nodes,
            function_kwargs=fit_function_kwargs,
            n_jobs=self.n_jobs,
            disable_progressbar=disable_progressbar, show_warnings=show_warnings,
            distributor=distributor
        )

        for model in fitted_models:
            if isinstance(model, tuple):
                self.hts_result.models = model
            else:
                self.hts_result.models = (model.node.key, model)
        return self

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

    def predict(self,
                exogenous_df: pandas.DataFrame = None,
                steps_ahead: int = None,
                distributor: Optional[DistributorBaseClass] = None,
                disable_progressbar=defaults.DISABLE_PROGRESSBAR,
                show_warnings=defaults.SHOW_WARNINGS,
                **predict_kwargs) -> pandas.DataFrame:
        """

        Parameters
        ----------
        distributor : Optional[DistributorBaseClass]
             A distributor, for parallel/distributed processing
        disable_progressbar : Bool
            Disable or enable progressbar
        show_warnings : Bool
            Disable warnings
        predict_kwargs : Any
            Any arguments to be passed to the underlying forecasting model's predict function
        exogenous_df : pandas.DataFrame
            A dataframe of lenght == steps_ahead containing the exogenous data for each of the nodes
        steps_ahead : int
            The number of forecasting steps for which to produce a forecast

        Returns
        -------
        Revised Forescasts, as a pandas.DataFrame in the same format as the one passed for fitting, extended by `steps_ahead`
        time steps`
        """

        steps_ahead = self.__init_predict_step(exogenous_df, steps_ahead)
        predict_function_kwargs = {'fit_kwargs': predict_kwargs,
                                   'steps_ahead': steps_ahead,
                                   'low_memory': self.low_memory,
                                   'tmp_dir': self.tmp_dir,
                                   'predict_kwargs': predict_kwargs,
                                   }

        fit_models = _model_mapping_to_iterable(self.hts_result.models, self.nodes)
        results = _do_predict(
            models=fit_models,
            function_kwargs=predict_function_kwargs,
            n_jobs=self.n_jobs,
            disable_progressbar=disable_progressbar,
            show_warnings=show_warnings,
            distributor=distributor

        )
        for key, forecast, error, residual in results:
            self.hts_result.forecasts = (key, forecast)
            self.hts_result.errors = (key, error)
            self.hts_result.residuals = (key, residual)
        return self._revise(steps_ahead=steps_ahead)

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

        freq = getattr(self.nodes.item.index, 'freq', 1)
        start = self.nodes.item.index[-1] + freq
        end = self.nodes.item.index[-1] + (steps_ahead * freq)

        future = pandas.date_range(freq=freq,
                                   start=start,
                                   end=end
                                   )

        return self.nodes.item.index.append(future)
