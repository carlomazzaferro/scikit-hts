from hts._t import ModelT
from hts.model.ar import AutoArimaModel, SarimaxModel
from hts.model.es import HoltWintersModel
from hts.model.p import FBProphetModel

__all__ = [
    "AutoArimaModel",
    "SarimaxModel",
    "HoltWintersModel",
    "FBProphetModel",
    "MODEL_MAPPING",
]


MODEL_MAPPING = {
    ModelT.auto_arima.name: AutoArimaModel,
    ModelT.holt_winters.name: HoltWintersModel,
    ModelT.prophet.name: FBProphetModel,
    ModelT.sarimax.name: SarimaxModel,
}
