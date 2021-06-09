from datetime import timedelta

import pandas

from hts import HTSRegressor
from hts.core.result import HTSResult


def test_instantiate_regressor():
    ht = HTSRegressor()
    assert ht.model == "prophet"
    assert isinstance(ht.hts_result, HTSResult)
    assert isinstance(ht.hts_result.residuals, dict)


def test_fit_regressor_uv(load_df_and_hier_uv):
    hierarchical_sine_data, sine_hier = load_df_and_hier_uv
    ht = HTSRegressor(model="holt_winters", revision_method="OLS")
    ht.fit(df=hierarchical_sine_data, nodes=sine_hier)
    assert isinstance(ht.hts_result.errors, dict)
    assert isinstance(ht.hts_result.models, dict)
    assert "total" in ht.hts_result.models


def test_predict_regressor_uv(load_df_and_hier_uv):
    hierarchical_sine_data, sine_hier = load_df_and_hier_uv
    hsd = hierarchical_sine_data.head(200)

    for model in ["holt_winters", "auto_arima", "sarimax"]:
        ht = HTSRegressor(model=model, revision_method="OLS")

        ht.fit(df=hsd, nodes=sine_hier)
        preds = ht.predict(steps_ahead=10)

        assert isinstance(preds, pandas.DataFrame)

        # base + steps ahead
        assert len(preds) == len(hsd) + 10
        assert len(ht.hts_result.forecasts["total"]) == len(hsd) + 10
        assert "total" in ht.hts_result.errors
        assert "total" in ht.hts_result.forecasts
        assert "total" in ht.hts_result.residuals

        assert "a" in ht.hts_result.errors
        assert "a" in ht.hts_result.forecasts
        assert "a" in ht.hts_result.residuals


def test_fit_regressor_visnights(load_df_and_hier_visnights):
    hierarchical_vis_data, vis_hier = load_df_and_hier_visnights
    ht = HTSRegressor(model="holt_winters", revision_method="OLS")
    ht.fit(df=hierarchical_vis_data, nodes=vis_hier)
    assert isinstance(ht.hts_result.errors, dict)
    assert isinstance(ht.hts_result.models, dict)
    assert "total" in ht.hts_result.models


def test_predict_regressor_visnights(load_df_and_hier_visnights):
    hierarchical_vis_data, vis_hier = load_df_and_hier_visnights
    hvd = hierarchical_vis_data
    for model in ["holt_winters", "auto_arima"]:
        ht = HTSRegressor(model=model, revision_method="OLS")

        ht.fit(df=hvd, nodes=vis_hier)
        preds = ht.predict(steps_ahead=4)

        assert isinstance(preds, pandas.DataFrame)

        # base + steps ahead
        assert len(preds) == len(hvd) + 4
        assert len(ht.hts_result.forecasts["total"]) == len(hvd) + 4
        assert "total" in ht.hts_result.errors
        assert "total" in ht.hts_result.forecasts
        assert "total" in ht.hts_result.residuals

        assert "NSW" in ht.hts_result.errors
        assert "NSW" in ht.hts_result.forecasts
        assert "NSW" in ht.hts_result.residuals


def test_exog_fit_predict_fb_model(hierarchical_mv_data, mv_tree_empty):
    exogenous = {
        k: ["precipitation", "temp"]
        for k in hierarchical_mv_data.columns
        if k not in ["precipitation", "temp"]
    }
    horizon = 7
    train = hierarchical_mv_data[:500]
    test = hierarchical_mv_data[500 : 500 + horizon]
    clf = HTSRegressor(model="prophet", revision_method="OLS", n_jobs=0)
    model = clf.fit(train, mv_tree_empty, exogenous=exogenous)
    preds = model.predict(exogenous_df=test)
    assert isinstance(preds, pandas.DataFrame)

    # base + steps ahead
    assert len(preds) == len(train) + horizon
    assert len(model.hts_result.forecasts["total"]) == len(train) + horizon

    assert len(model.hts_result.errors) == len(train.columns) - 2

    for column in train.columns:
        if column not in ("precipitation", "temp"):
            assert column in model.hts_result.errors
            assert column in model.hts_result.forecasts
            assert column in model.hts_result.residuals
