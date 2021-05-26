from typing import Dict, Optional, Tuple

import pandas


class HTSResult:
    def __init__(self) -> None:
        self._revised_forecasts: Optional[Dict] = None
        self._models: Dict = dict()
        self._errors: Dict = dict()
        self._residuals: Dict = dict()
        self._forecasts: Dict = dict()

    @property
    def forecasts(self) -> Dict:
        return self._forecasts

    @forecasts.setter
    def forecasts(self, kv_tuple: Tuple) -> None:
        k, v = kv_tuple
        self._forecasts[k] = v

    @property
    def errors(self) -> Dict:
        return self._errors

    @errors.setter
    def errors(self, kv_tuple: Tuple) -> None:
        k, v = kv_tuple
        self._errors[k] = v

    @property
    def residuals(self) -> Dict:
        return self._residuals

    @residuals.setter
    def residuals(self, kv_tuple: Tuple) -> None:
        k, v = kv_tuple
        self._residuals[k] = v

    @property
    def models(self) -> Dict:
        return self._models

    @models.setter
    def models(self, kv_tuple: Tuple) -> None:
        k, v = kv_tuple
        self._models[k] = v
