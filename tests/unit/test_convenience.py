import numpy
import pandas
import pytest

from hts import HTSRegressor
from hts.convenience import (
    _calculate_errors,
    _sanitize_errors_dict,
    _sanitize_forecasts_dict,
    revise_forecasts,
)


def test_convenience_error_handling(load_df_and_hier_uv):
    _, _ = load_df_and_hier_uv
    dummy = numpy.array([1, 2, 3, 5])
    with pytest.raises(ValueError):
        revise_forecasts("FP", forecasts={"a": dummy}, nodes=None)

    with pytest.raises(ValueError):
        revise_forecasts("OLS", summing_matrix=dummy, forecasts={"a": dummy})


def test_errors_and_residuals_dicts():
    res_a, res_b = [1, 2, 1, 3, 1], [0, 0, 1, 0, 0]
    error_a, error_b = 0.1, 0.999
    errors = {"a": error_a, "b": error_b}
    residuals = {"a": numpy.array(res_a), "b": numpy.array(res_b)}
    assert _sanitize_errors_dict(errors) == errors
    assert _calculate_errors(method="A", errors=errors) == errors
    calculated_errors_from_res = _calculate_errors(method="A", residuals=residuals)
    assert isinstance(calculated_errors_from_res, dict)
    assert (
        _calculate_errors(method="A", errors=errors, residuals=residuals)
        == calculated_errors_from_res
    )
    assert calculated_errors_from_res["a"] == numpy.mean(numpy.array(res_a) ** 2)
    assert calculated_errors_from_res["b"] == numpy.mean(numpy.array(res_b) ** 2)

    with pytest.raises(ValueError):
        _sanitize_errors_dict(errors={"a": "12340"})  # pragma: nocover


def test_sanitize_forecasts():
    values = [[1, 2, 1, 3, 1], [0, 0, 0, 0, 0]]
    forecasts = {"a": numpy.array(values[0]), "b": numpy.array(values[1])}
    sanitized = _sanitize_forecasts_dict(forecasts)
    assert isinstance(sanitized, dict)
    assert isinstance(sanitized["a"].yhat, pandas.Series)

    forecasts = {"a": pandas.Series(values[0]), "b": pandas.Series(values[1])}
    sanitized = _sanitize_forecasts_dict(forecasts)
    assert isinstance(sanitized, dict)
    assert isinstance(sanitized["a"].yhat, pandas.Series)

    forecasts = {"a": pandas.DataFrame({"1": values[0]})}
    sanitized = _sanitize_forecasts_dict(forecasts)
    assert isinstance(sanitized, dict)
    assert isinstance(sanitized["a"].yhat, pandas.Series)

    forecasts = {"a": numpy.array(values), "b": numpy.array(values)}
    with pytest.raises(ValueError):
        _sanitize_forecasts_dict(forecasts)

    forecasts = {"a": pandas.DataFrame({"1": values[0], "2": values[1]})}
    with pytest.raises(ValueError):
        _sanitize_forecasts_dict(forecasts)


def test_convenience(load_df_and_hier_uv):
    hd, hier = load_df_and_hier_uv
    hd = hd.head(200)
    ht = HTSRegressor(model="holt_winters", revision_method="OLS")
    ht = ht.fit(df=hd, nodes=hier)
    preds = ht.predict(steps_ahead=10)

    rev = revise_forecasts(
        "OLS",
        nodes=ht.nodes,
        forecasts=ht.hts_result.forecasts,
        errors=ht.hts_result.errors,
        residuals=ht.hts_result.residuals,
    )
    assert isinstance(rev, pandas.DataFrame)
    assert numpy.allclose(preds.values, rev.values)

    rev_s = revise_forecasts(
        "OLS",
        summing_matrix=ht.sum_mat,
        forecasts=ht.hts_result.forecasts,
        errors=ht.hts_result.errors,
        residuals=ht.hts_result.residuals,
    )
    assert isinstance(rev_s, pandas.DataFrame)
    assert numpy.allclose(rev.values, rev_s.values)

    rev = revise_forecasts("AHP", nodes=ht.nodes, forecasts=ht.hts_result.forecasts)
    assert isinstance(rev, pandas.DataFrame)
