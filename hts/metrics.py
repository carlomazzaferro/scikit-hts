import numpy
import pandas

__all__ = ['mean_absolute_scaled_error']

def mean_absolute_scaled_error(y_true, y_pred, y_train,
                              naive_period=1,
                              sample_weight=None,
                              multioutput='raw_values'):
    
    """Mean absolute scaled error


    Parameters
    ----------
    y_true : pandas.DataFrame of shape (n_samples,) or (n_samples, n_outputs)
        Observed test values of y.

    y_pred : pandas.DataFrame of shape (n_samples,) or (n_samples, n_outputs)
        Predicted / Forecasted values of y.
    
    y_train : pandas.DataFrame of shape (n_samples,) or (n_samples, n_outputs)
        Observed train values of y.

    multioutput : string in ['raw_values', 'uniform_average']
        raw_values: Returns metric for each node in dataset.
        unfirom_average: Returns average value for all nodes.

    Returns
    --------
    loss : 

    """
    
    in_sample_naive_forecast_errors = numpy.abs(
        y_train.diff(naive_period)[naive_period:]
        )
    
    in_sample_mae = in_sample_naive_forecast_errors.apply(numpy.average)

    e_t = numpy.abs(y_true - y_pred)
    q_t = (e_t / in_sample_mae)
    
    output_errors = q_t.apply(numpy.average, axis=0, weights=sample_weight)

    if isinstance(multioutput, str):
        if multioutput=='raw_values':
            return output_errors
        elif multioutput=='uniform_average':
            multioutput = None

    output_errors_avg = numpy.average(output_errors, weights=multioutput)

    return output_errors_avg