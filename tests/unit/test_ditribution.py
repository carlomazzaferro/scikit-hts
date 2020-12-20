# -*- coding: utf-8 -*-
# Many thanks to @blue-yonder for providing the base implementation for this file.
# see more at: https://github.com/blue-yonder/tsfresh

from itertools import chain

import numpy as np
import pytest
from distributed import Client

from hts import HTSRegressor
from hts.utilities.distribution import (
    ClusterDaskDistributor,
    LocalDaskDistributor,
    MultiprocessingDistributor,
)


def test_partition():
    distributor = MultiprocessingDistributor(n_workers=1)

    data = [1, 3, 10, -10, 343.0]
    distro = distributor.partition(data, 3)
    assert next(distro), [1, 3, 10]
    assert next(distro), [-10, 343.0]

    data = np.arange(10)
    distro = distributor.partition(data, 2)
    assert next(distro), [0, 1]
    assert next(distro), [2, 3]


@pytest.mark.serial
def test_calculate_best_chunk_size():
    distributor = MultiprocessingDistributor(n_workers=2)
    assert distributor.calculate_best_chunk_size(10), 1
    assert distributor.calculate_best_chunk_size(11), 2
    assert distributor.calculate_best_chunk_size(100), 10
    assert distributor.calculate_best_chunk_size(101), 11

    distributor = MultiprocessingDistributor(n_workers=3)
    assert distributor.calculate_best_chunk_size(10), 1
    assert distributor.calculate_best_chunk_size(30), 2
    assert distributor.calculate_best_chunk_size(31), 3


#


@pytest.mark.serial
def test_multiprocessing_fit(load_df_and_hier_uv):
    distributor = MultiprocessingDistributor(n_workers=2)
    hsd, hier = load_df_and_hier_uv
    reg = HTSRegressor()
    reg = reg.fit(df=hsd, nodes=hier, distributor=distributor)

    node_keys = list(chain.from_iterable([v for v in hier.values()]))

    for node in node_keys:
        assert node in reg.hts_result.models.keys()


@pytest.mark.serial
def test_dask_cluster_two_workers(load_df_and_hier_uv):
    with Client(n_workers=1, processes=False) as client:
        address = client.scheduler_info()["address"]
        distributor = ClusterDaskDistributor(address=address)

        hsd, hier = load_df_and_hier_uv
        reg = HTSRegressor()
        reg = reg.fit(df=hsd, nodes=hier, distributor=distributor)

        node_keys = list(chain.from_iterable([v for v in hier.values()]))

        for node in node_keys:
            assert node in reg.hts_result.models.keys()
