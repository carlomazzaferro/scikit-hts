from collections import namedtuple

import numpy
import pandas
import pytest
from fbprophet import Prophet
from pmdarima import AutoARIMA

from hts.model import AutoArimaModel, FBProphetModel, HoltWintersModel, SarimaxModel
from hts.model.base import TimeSeriesModel


def test_instantiate_fb_model_uv(uv_tree):
    fb = FBProphetModel(node=uv_tree)
    assert isinstance(fb, TimeSeriesModel)
    fb = FBProphetModel(node=uv_tree, capacity_max=1)
    assert isinstance(fb, TimeSeriesModel)
    fb = FBProphetModel(node=uv_tree, capacity_min=1)
    assert isinstance(fb, TimeSeriesModel)


def test_fit_predict_fb_model_mv(mv_tree):
    exog = pandas.DataFrame({"precipitation": [1], "temp": [20]})
    fb = FBProphetModel(node=mv_tree)
    assert isinstance(fb, TimeSeriesModel)
    fb.fit()
    fb.predict(mv_tree, exogenous_df=exog)
    assert isinstance(fb.forecast, pandas.DataFrame)
    assert isinstance(fb.residual, numpy.ndarray)
    assert isinstance(fb.mse, float)


def test_fit_predict_fb_model_mv(mv_tree):
    exog = pandas.DataFrame({"precipitation": [1, 2], "temp": [20, 30]})
    fb = FBProphetModel(node=mv_tree)
    assert isinstance(fb, TimeSeriesModel)
    fb.fit()
    fb.predict(mv_tree, exogenous_df=exog)
    assert isinstance(fb.forecast, pandas.DataFrame)
    assert isinstance(fb.residual, numpy.ndarray)
    assert isinstance(fb.mse, float)


def test_fit_predict_fb_model_uv(uv_tree):
    fb = FBProphetModel(node=uv_tree)
    fb.fit()
    assert isinstance(fb.model, Prophet)
    fb.predict(uv_tree)
    assert isinstance(fb.forecast, pandas.DataFrame)
    assert isinstance(fb.residual, numpy.ndarray)
    assert isinstance(fb.mse, float)


def test_fit_predict_ar_model_mv(mv_tree):
    ar = AutoArimaModel(node=mv_tree)
    ar.fit(max_iter=1)
    assert isinstance(ar.model, AutoARIMA)
    exog = pandas.DataFrame({"precipitation": [1], "temp": [20]})
    ar.predict(mv_tree, steps_ahead=1, exogenous_df=exog)
    assert isinstance(ar.forecast, pandas.DataFrame)
    assert isinstance(ar.residual, numpy.ndarray)
    assert isinstance(ar.mse, float)


def test_fit_predict_ar_model_uv(uv_tree):
    ar = AutoArimaModel(
        node=uv_tree,
    )
    ar.fit(max_iter=1)
    assert isinstance(ar.model, AutoARIMA)
    ar.predict(uv_tree)
    assert isinstance(ar.forecast, pandas.DataFrame)
    assert isinstance(ar.residual, numpy.ndarray)
    assert isinstance(ar.mse, float)


def test_fit_predict_sarimax_model_uv(uv_tree):
    sar = SarimaxModel(
        node=uv_tree,
        max_iter=1,
    )
    fitted_sar = sar.fit()
    assert isinstance(fitted_sar, SarimaxModel)
    sar.predict(uv_tree)
    assert isinstance(sar.forecast, pandas.DataFrame)
    assert isinstance(sar.residual, numpy.ndarray)
    assert isinstance(sar.mse, float)


def test_fit_predict_hw_model_uv(uv_tree):
    hw = HoltWintersModel(
        node=uv_tree,
    )
    fitted_hw = hw.fit()
    assert isinstance(fitted_hw, HoltWintersModel)
    hw.predict(uv_tree)
    assert isinstance(hw.forecast, pandas.DataFrame)
    assert isinstance(hw.residual, numpy.ndarray)
    assert isinstance(hw.mse, float)


def test_fit_predict_hw_model_uv_with_transform(uv_tree):
    Transform = namedtuple("Transform", ["func", "inv_func"])
    transform_pos_neg = Transform(func=numpy.exp, inv_func=lambda x: -x)

    hw = HoltWintersModel(node=uv_tree, transform=transform_pos_neg)
    fitted_hw = hw.fit()
    assert isinstance(fitted_hw, HoltWintersModel)
    preds = hw.predict(uv_tree)
    assert not (preds.forecast.values > 0).any()

    assert isinstance(hw.forecast, pandas.DataFrame)
    assert isinstance(hw.residual, numpy.ndarray)
    assert isinstance(hw.mse, float)


def test_fit_predict_model_invalid_transform(uv_tree):
    Transform = namedtuple("Transform", ["func_invalid_arg", "inv_func"])
    transform_pos_neg = Transform(func_invalid_arg=numpy.exp, inv_func=lambda x: -x)
    with pytest.raises(ValueError):
        HoltWintersModel(node=uv_tree, transform=transform_pos_neg)
