import pandas

from hts import HTSRegressor
from hts.core.result import HTSResult


def test_instantiate_regressor():
    ht = HTSRegressor()
    assert ht.model == 'prophet'
    assert isinstance(ht.hts_result, HTSResult)
    assert isinstance(ht.hts_result.residuals, dict)


def test_fit_regressor_uv(load_df_and_hier_uv):
    hierarchical_sine_data, sine_hier = load_df_and_hier_uv
    ht = HTSRegressor(model='holt_winters', revision_method='OLS')
    ht.fit(df=hierarchical_sine_data, nodes=sine_hier)
    assert isinstance(ht.hts_result.errors, dict)
    assert isinstance(ht.hts_result.models, dict)
    assert 'total' in ht.hts_result.models


def test_predict_regressor_uv(load_df_and_hier_uv):
    hierarchical_sine_data, sine_hier = load_df_and_hier_uv
    hsd = hierarchical_sine_data.head(200)

    for model in ['holt_winters', 'auto_arima', 'sarimax']:
        ht = HTSRegressor(model=model, revision_method='OLS')

        ht.fit(df=hsd, nodes=sine_hier)
        preds = ht.predict(steps_ahead=10)

        assert isinstance(preds, pandas.DataFrame)

        # base + steps ahead
        assert len(preds) == len(hsd) + 10
        assert len(ht.hts_result.forecasts['total']) == len(hsd) + 10
        assert 'total' in ht.hts_result.errors
        assert 'total' in ht.hts_result.forecasts
        assert 'total' in ht.hts_result.residuals

        assert 'a' in ht.hts_result.errors
        assert 'a' in ht.hts_result.forecasts
        assert 'a' in ht.hts_result.residuals


def test_fit_regressor_visnights(load_df_and_hier_visnights):
    hierarchical_vis_data, vis_hier = load_df_and_hier_visnights
    ht = HTSRegressor(model='holt_winters', revision_method='OLS')
    ht.fit(df=hierarchical_vis_data, nodes=vis_hier)
    assert isinstance(ht.hts_result.errors, dict)
    assert isinstance(ht.hts_result.models, dict)
    assert 'total' in ht.hts_result.models


def test_predict_regressor_visnights(load_df_and_hier_visnights):
    hierarchical_vis_data, vis_hier = load_df_and_hier_visnights
    hvd = hierarchical_vis_data
    for model in ['holt_winters', 'auto_arima']:
        ht = HTSRegressor(model=model, revision_method='OLS')

        ht.fit(df=hvd, nodes=vis_hier)
        preds = ht.predict(steps_ahead=4)

        assert isinstance(preds, pandas.DataFrame)

        # base + steps ahead
        assert len(preds) == len(hvd) + 4
        assert len(ht.hts_result.forecasts['total']) == len(hvd) + 4
        assert 'total' in ht.hts_result.errors
        assert 'total' in ht.hts_result.forecasts
        assert 'total' in ht.hts_result.residuals

        assert 'NSW' in ht.hts_result.errors
        assert 'NSW' in ht.hts_result.forecasts
        assert 'NSW' in ht.hts_result.residuals

