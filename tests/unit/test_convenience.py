import numpy
import pandas
import pytest

from hts import HTSRegressor
from hts.convenience import revise_forecasts


def test_convenience_error_handling(load_df_and_hier_uv):
    hd, hier = load_df_and_hier_uv
    dummy = numpy.array([1, 2, 3, 5])
    with pytest.raises(ValueError):
        revise_forecasts('FP', forecasts={'a': dummy}, nodes=None)

    with pytest.raises(ValueError):
        revise_forecasts('OLS',
                         summing_matrix=dummy,
                         forecasts={'a': dummy})


def test_convenience(load_df_and_hier_uv):
    hd, hier = load_df_and_hier_uv
    hd = hd.head(200)
    ht = HTSRegressor(model='holt_winters', revision_method='OLS')
    ht = ht.fit(df=hd, nodes=hier)
    preds = ht.predict(steps_ahead=10)

    rev = revise_forecasts(
        'OLS',
        nodes=ht.nodes,
        forecasts=ht.hts_result.forecasts,
        errors=ht.hts_result.errors,
        residuals=ht.hts_result.residuals,
    )

    assert isinstance(rev, pandas.DataFrame)
    assert numpy.allclose(preds.values, rev.values)

    rev_s = revise_forecasts(
        'OLS',
        summing_matrix=ht.sum_mat,
        forecasts=ht.hts_result.forecasts,
        errors=ht.hts_result.errors,
        residuals=ht.hts_result.residuals,
    )
    assert isinstance(rev_s, pandas.DataFrame)
    assert numpy.allclose(rev.values, rev_s.values)

    rev = revise_forecasts(
        'AHP',
        nodes=ht.nodes,
        forecasts=ht.hts_result.forecasts
    )
    assert isinstance(rev, pandas.DataFrame)

