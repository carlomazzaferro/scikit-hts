import numpy
from hts.metrics import mean_absolute_scaled_error


def test_mase_sales_example(sales_example_data):

    y_train = sales_example_data['y_train']
    y_test = sales_example_data['y_test']
    y_pred = sales_example_data['y_pred']

    mase_error = mean_absolute_scaled_error(y_test, y_pred, y_train)
    expected = 0.1964286

    assert numpy.allclose(expected, mase_error)

def test_mase_visnights_y_pred_equals_y_true(hierarchical_visnights_data):
    data = hierarchical_visnights_data

    y_train = data.iloc[:6, :]
    y_true = data.iloc[6:, :]
    y_pred = data.iloc[6:, :]

    error = mean_absolute_scaled_error(y_true, 
                                       y_pred, 
                                       y_train, 
                                       multioutput='raw_values')

    assert numpy.all(error == 0)

def test_mase_multioutput_format(hierarchical_visnights_data):
    data = hierarchical_visnights_data

    y_train = data.iloc[:6, :]
    y_true = data.iloc[6:, :]
    y_pred = data.iloc[6:, :]

    error_raw = mean_absolute_scaled_error(y_true, 
                                           y_pred, 
                                           y_train, 
                                           multioutput='raw_values')

    error_avg = mean_absolute_scaled_error(y_true, 
                                           y_pred, 
                                           y_train, 
                                           multioutput='uniform_average')

    assert numpy.shape(error_raw) == (27,)
    assert isinstance(error_avg, float)