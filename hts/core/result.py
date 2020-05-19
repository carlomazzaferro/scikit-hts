import pandas


class HTSResult:

    def __init__(self):
        self._revised_forecasts = None
        self._models = dict()
        self._errors = dict()
        self._residuals = dict()
        self._forecasts = dict()

    @property
    def forecasts(self):
        return self._forecasts

    @forecasts.setter
    def forecasts(self, kv_tuple):
        k, v = kv_tuple
        self._forecasts[k] = v

    @property
    def errors(self):
        return self._errors

    @errors.setter
    def errors(self, kv_tuple):
        k, v = kv_tuple
        self._errors[k] = v

    @property
    def residuals(self):
        return self._residuals

    @residuals.setter
    def residuals(self, kv_tuple):
        k, v = kv_tuple
        self._residuals[k] = v

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, kv_tuple):
        k, v = kv_tuple
        self._models[k] = v

    def to_pandas(self, p):
        kv = getattr(self, p)
        return pandas.DataFrame(kv)

    def get_series(self, p, key):
        prop = getattr(self, p)
        return prop[key]
