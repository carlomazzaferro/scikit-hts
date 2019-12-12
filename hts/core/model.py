import pandas

from hts.core.types import NAryTreeT


class HierarchicalModel:

    def __init__(self, nodes: NAryTreeT):
        self._transformers
        self.nodes = nodes


class Forecast:

    def __init__(self, df: pandas.DataFrame):
        self.df = df

    @property
    def yhat(self):
        return self.df.yhat

