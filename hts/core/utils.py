from hts.utilities.distribution import MapDistributor, MultiprocessingDistributor, DistributorBaseClass


def _do_fit(models,
            fit_kwargs,
            n_jobs,
            disable_progressbar,
            show_warnings,
            distributor):

    if distributor is None:
        if n_jobs == 0:
            distributor = MapDistributor(disable_progressbar=disable_progressbar,
                                         progressbar_title="Feature Extraction")
        else:
            distributor = MultiprocessingDistributor(n_workers=n_jobs,
                                                     disable_progressbar=disable_progressbar,
                                                     progressbar_title="Feature Extraction",
                                                     show_warnings=show_warnings)

    if not isinstance(distributor, DistributorBaseClass):
        raise ValueError("the passed distributor is not an DistributorBaseClass object")

    result = distributor.map_reduce(_do_actual_fit,
                                    data=models,
                                    function_kwargs=fit_kwargs)
    distributor.close()
    return result


def _do_actual_fit(node_model, fit_kwargs):
    model_instance = node_model.fit(**fit_kwargs)
    return model_instance

