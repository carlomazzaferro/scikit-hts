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

