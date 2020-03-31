import numpy
import pandas
import pytest

from hts.transforms import FunctionTransformer


def test_fit_transform(hierarchical_mv_data):
    tf = FunctionTransformer()
    df = hierarchical_mv_data
    transformed = tf.fit_transform(df['WF-01'])
    assert isinstance(transformed, numpy.ndarray)
    assert tf.lam is not None

    transformed = tf.fit_transform(df['WF-01'].replace(0, 6))
    assert isinstance(transformed, numpy.ndarray)
    assert tf.lam is not None

    with pytest.raises(ValueError):
        _ = tf.fit_transform(df['precipitation'].replace(0, -10))


def test_inv_transform(hierarchical_mv_data):
    tf = FunctionTransformer()
    df = hierarchical_mv_data

    base = df['WF-01'].replace(0, 6)
    transformed = tf.fit_transform(base)

    inv_transformed = tf.inverse_transform(transformed)
    assert numpy.allclose(inv_transformed, base, atol=0.1)

    inv_transformed = tf.inverse_transform(pandas.Series(transformed))
    assert numpy.allclose(inv_transformed, base, atol=0.1)
