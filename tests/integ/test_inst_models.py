import numpy
import pandas
from fbprophet import Prophet

from hts import FBProphetModel
from hts.model import TimeSeriesModel


def test_instantiate_fb_model_uv(uv_tree):
    fb = FBProphetModel(
        node=uv_tree
    )
    assert isinstance(fb, TimeSeriesModel)
    fb = FBProphetModel(
        node=uv_tree,
        capacity_max=1
    )
    assert isinstance(fb, TimeSeriesModel)
    fb = FBProphetModel(
        node=uv_tree,
        capacity_min=1
    )
    assert isinstance(fb, TimeSeriesModel)


def test_instantiate_fb_model_mv(mv_tree):
    fb = FBProphetModel(
        node=mv_tree
    )
    assert isinstance(fb, TimeSeriesModel)


def test_fit_predict_fb_model_uv(uv_tree):
    fb = FBProphetModel(
        node=uv_tree
    )
    fb.fit_predict()
    assert isinstance(fb.model, Prophet)
    assert isinstance(fb.forecast, pandas.DataFrame)
    assert isinstance(fb.residual, numpy.ndarray)
    assert isinstance(fb.mse, float)



