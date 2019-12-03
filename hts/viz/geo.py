import string

import numpy
from h3 import h3

from hts import logger
from hts.types import NAryTreeT, HierarchyVisualizerT
from hts.helpers import flatten


def get_min_max_ll(geos):
    fl = list(flatten([g[0] for g in geos]))
    mx_lat = max([x[0] for x in fl])
    mx_lon = max([x[1] for x in fl])

    mn_lat = min([x[0] for x in fl])
    mn_lon = min([x[1] for x in fl])

    return mx_lat, mx_lon, mn_lat, mn_lon


class HierarchyVisualizer(HierarchyVisualizerT):
    def __init__(self, tree: NAryTreeT):
        self.tree = tree

    @property
    def as_df(self):
        return self.tree.to_pandas()

    def get_geos(self):
        h3s = [col for col in self.as_df.columns if all(c in string.hexdigits for c in col)]
        return [(h3.h3_to_geo_boundary(g), self.as_df[g].fillna(0).sum(), g) for g in h3s]

    def h3_to_lat_long(self):
        return

    def create_map(self):
        try:
            from folium import Map
            from folium.vector_layers import Polygon
        except ImportError:
            logger.error('Mapping requires folium==0.10.0 to be installed')
            return

        _map = Map(tiles="cartodbpositron")
        geos = self.get_geos()
        max_lat, max_lon, min_lat, min_lon = get_min_max_ll(geos)

        geos = [(i, numpy.log(j), k) for i, j, k in geos]
        mx, mn = max([j for i, j, k in geos]), min([j for i, j, k in geos])
        geos = [(i, (j - mn) / (mx - mn), k) for i, j, k in geos]

        for points, count, h in geos:
            tooltip = f"hex: {h}"
            polygon = Polygon(locations=points,
                              tooltip=tooltip,
                              fill=True,
                              color='#ff0000',
                              fill_color='#ff0000',
                              fill_opacity=count,
                              weight=3,
                              opacity=0.4)
            polygon.add_to(_map)

        _map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

        return _map
