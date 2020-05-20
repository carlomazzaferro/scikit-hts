import numpy

from hts import HTSRegressor
from hts.revision import RevisionMethod


def test_instantiate_revision(load_df_and_hier_uv):
    hierarchical_sine_data, sine_hier = load_df_and_hier_uv
    hsd = hierarchical_sine_data.head(200)

    for method in ['OLS', 'FP', 'WLSS', 'WLSV', 'PHA', 'AHP', 'NONE']:
        ht = HTSRegressor(model='holt_winters', revision_method=method)
        ht.fit(df=hsd, nodes=sine_hier)

        rm = RevisionMethod(method, sum_mat=ht.sum_mat, transformer=ht.transform)

        _ = ht.predict(steps_ahead=3)
        revised = rm.revise(
            forecasts=ht.hts_result.forecasts,
            mse=ht.hts_result.errors,
            nodes=ht.nodes
        )
        assert isinstance(revised, numpy.ndarray)
        assert revised.shape == (203, len(ht.hts_result.forecasts))
