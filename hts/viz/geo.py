import logging
import string
from itertools import chain

import numpy

from hts._t import HierarchyVisualizerT, NAryTreeT

logger = logging.getLogger(__name__)


def get_min_max_ll(geos):
    fl = list(chain.from_iterable([g[0] for g in geos]))
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
        try:
            from h3 import h3
        except ImportError:  # pragma: no cover
            logger.error(
                "h3-py must be installed for geo hashing capabilities. Exiting."
                "Install it with: pip install scikit-hts[geo]"
            )
            return
        h3s = [
            col for col in self.as_df.columns if all(c in string.hexdigits for c in col)
        ]
        return [
            (h3.h3_to_geo_boundary(g), self.as_df[g].fillna(0).sum(), g) for g in h3s
        ]

    def h3_to_lat_long(self):
        return

    def create_map(self):

        try:
            import branca.colormap as cm
            from folium import Map
            from folium.vector_layers import Polygon
        except ImportError:  # pragma: no cover
            logger.error(
                "Mapping requires folium==0.10.0 to be installed, geo mapping will not work."
                "Install it with: pip install scikit-hts[geo]"
            )
            return

        _map = Map(tiles="cartodbpositron")
        geos = self.get_geos()
        max_lat, max_lon, min_lat, min_lon = get_min_max_ll(geos)

        geos = [
            (i, numpy.log(j + 1) / (self.tree.get_node_height(k) + 1), k)
            for i, j, k in geos
        ]
        mx, mn = max([j for i, j, k in geos]), min([j for i, j, k in geos])
        geos = [(i, (j - mn) / (mx - mn), k) for i, j, k in geos]

        for points, count, h in sorted(geos, key=lambda x: x[1]):
            tooltip = f"hex: {h}"
            polygon = Polygon(
                locations=points,
                tooltip=tooltip,
                fill=True,
                color=cm.linear.OrRd_03.rgb_hex_str(count),
                fill_color=cm.linear.OrRd_03.rgb_hex_str(count),
                fill_opacity=0.3,
                weight=3,
                opacity=0.4,
            )
            polygon.add_to(_map)

        _map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

        return _map
