import numpy
from folium import Map
from folium.vector_layers import Polygon


def create_map(geos):
    # Create the map object
    _map = Map(tiles="cartodbpositron")

    scaled = [numpy.log(x[1]) for x in geos]
    mx, mn = max(scaled), min(scaled)
    rescaled = [(x - mn) / (mx - mn) for x in scaled]
    for i, g in enumerate(geos):
        points = g[0]
        tooltip = f"hex: {g[2]}"
        polygon = Polygon(locations=points,
                          tooltip=tooltip,
                          fill=True,
                          color='#ff0000',
                          fill_color='#ff0000',
                          fill_opacity=rescaled[i],
                          weight=3,
                          opacity=0.4)
        polygon.add_to(_map)

    t = [g[0] for g in geos][0]
    max_lat = max([k[0] for k in t])
    min_lat = min([k[0] for k in t])

    min_lon = max([k[1] for k in t])
    max_lon = min([k[1] for k in t])

    # Fit the map to the bounds
    _map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    return _map
