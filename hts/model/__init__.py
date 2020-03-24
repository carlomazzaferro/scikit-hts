from hts._t import Model

from hts.model.ar import AutoArimaModel, SarimaxModel
from hts.model.es import HoltWintersModel
from hts.model.p import FBProphetModel

__all__ = ['AutoArimaModel',
           'SarimaxModel',
           'HoltWintersModel',
           'FBProphetModel',
           'MODEL_MAPPING']


MODEL_MAPPING = {
    Model.auto_arima.name: AutoArimaModel,
    Model.holt_winters.name: HoltWintersModel,
    Model.prophet.name: FBProphetModel,
    Model.sarimax.name: SarimaxModel
}