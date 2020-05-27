import numpy
import pandas

__all__ = ['mean_absolute_scaled_error']

def mean_absolute_scaled_error(y_true, y_pred, y_train,
                              m=1,
                              sample_weight=None,
                              multioutput='raw_values'):
    
    """Compute the Mean Absolute Scaled Error.

    The Mean Absolute Scaled Error (MASE) is an error metric that is
    scale independent and symmetric. It was proposed by
    Hyndman and Koehler (2006) as a generally applicable measurement 
    of forecast for comparing forecast errors in multiple time series, such as
    hierarchical ones.

    The error is scaled on the in-sample MAE (Mean Absolute Error) from
    naive forecast method. The metric is defined as:

        :math: `\frac{\frac{1}{J}\sum_{j}\left| y_{t+j} - \hat{y}_{t+j} \right|}{
            \frac{1}{T-m}\sum_{t=m+1}^T \left| y_t - \hat{y}_{t-m}\right|}`

    Where a perfect score is 0.0. Scores higher than one indicate that 
    model performs systematically worse than a naive forecast. Scores between
    0 and 1 performs better than a naive method.

    Parameters
    ----------
    y_true : pandas.DataFrame of shape (n_samples,) or (n_samples, n_outputs)
        Observed test values of y.

    y_pred : pandas.DataFrame of shape (n_samples,) or (n_samples, n_outputs)
        Predicted / Forecasted values of y.
    
    y_train : pandas.DataFrame of shape (n_samples,) or (n_samples, n_outputs)
        Observed train values of y.
    
    m : int
        An integer value representing number of lags used to calculate in-sample
        seasonal error. For example, daily seasonal data we use m = 7, for
        no seasonality, m = 1 (default).

    multioutput : string in ['raw_values', 'uniform_average']
        raw_values: Returns metric for each node in dataset.
        unfirom_average: Returns average value for all nodes.

    Returns
    --------
    loss : float or pandas.Series.
        if multioutput is 'raw_values', then mase is returned for each node
        in the hierarchical time series.
        if multioutput is 'uniform_average', then an average value for all
        nodes is returned.
    """
    
    in_sample_naive_forecast_errors = numpy.abs(
        y_train.diff(m)[m:]
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