from collections import namedtuple

import numpy
import pandas
import pytest

from hts.transforms import BoxCoxTransformer, FunctionTransformer


def test_fit_transform(hierarchical_mv_data):
    tf = BoxCoxTransformer()
    df = hierarchical_mv_data
    transformed = tf.fit_transform(df["WF-01"])
    assert isinstance(transformed, numpy.ndarray)
    assert tf.lam is not None

    transformed = tf.fit_transform(df["WF-01"].replace(0, 6))
    assert isinstance(transformed, numpy.ndarray)
    assert tf.lam is not None

    with pytest.raises(ValueError):
        _ = tf.fit_transform(df["precipitation"].replace(0, -10))


def test_inv_transform(hierarchical_mv_data):
    tf = BoxCoxTransformer()
    df = hierarchical_mv_data

    base = df["WF-01"].replace(0, 6)
    transformed = tf.fit_transform(base)

    inv_transformed = tf.inverse_transform(transformed)
    assert numpy.allclose(inv_transformed, base, atol=0.1)

    inv_transformed = tf.inverse_transform(pandas.Series(transformed))
    assert numpy.allclose(inv_transformed, base, atol=0.1)


def test_custom_transform(hierarchical_mv_data):
    def scale(val, src, dst):
        return ((val - src[0]) / (src[1] - src[0])) * (dst[1] - dst[0]) + dst[0]

    df = hierarchical_mv_data
    values = df["WF-01"]
    values_centered = scale(values, [min(values), max(values)], [0, 1])
    Transform = namedtuple("Transform", ["func", "inv_func"])

    transform_pos_neg = Transform(func=numpy.exp, inv_func=lambda x: -x)
    transform_isomorphic = Transform(func=numpy.exp, inv_func=numpy.log)

    function_transform_pos_neg = FunctionTransformer(
        func=getattr(transform_pos_neg, "func"),
        inv_func=getattr(transform_pos_neg, "inv_func"),
    )
    function_transform_isomorphic = FunctionTransformer(
        func=getattr(transform_isomorphic, "func"),
        inv_func=getattr(transform_isomorphic, "inv_func"),
    )

    transformed = function_transform_pos_neg.fit_transform(values_centered)
    assert isinstance(transformed, numpy.ndarray)
    assert not (transformed < 0).any()

    inv_transformed = function_transform_pos_neg.inverse_transform(values_centered)
    assert isinstance(inv_transformed, pandas.Series)
    assert not (inv_transformed.values > 0).any()

    transformed = function_transform_isomorphic.fit_transform(values_centered)
    inv_transformed = function_transform_isomorphic.inverse_transform(
        pandas.Series(transformed)
    )
    assert numpy.allclose(inv_transformed, values_centered, atol=0.1)
