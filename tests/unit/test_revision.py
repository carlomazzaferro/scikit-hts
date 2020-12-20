import numpy

from hts import HTSRegressor
from hts.revision import RevisionMethod


def test_instantiate_revision(load_df_and_hier_visnights):
    hierarchical_visnights_data, visnights_hier = load_df_and_hier_visnights

    for method in ["OLS", "FP", "WLSS", "WLSV", "PHA", "AHP", "NONE"]:
        ht = HTSRegressor(model="holt_winters", revision_method=method)
        ht.fit(df=hierarchical_visnights_data, nodes=visnights_hier)

        rm = RevisionMethod(method, sum_mat=ht.sum_mat, transformer=ht.transform)

        _ = ht.predict(steps_ahead=3)
        revised = rm.revise(
            forecasts=ht.hts_result.forecasts, mse=ht.hts_result.errors, nodes=ht.nodes
        )
        assert isinstance(revised, numpy.ndarray)
        assert revised.shape == (11, len(ht.hts_result.forecasts))
