from owslib.wfs import WebFeatureService
import geojson
import geopandas as gpd
from shapely import geometry
def get_wfs_data(url: str, type_name: str, box, srs) -> gpd.GeoDataFrame | None:
    # Specify the url for the backend.
    wfs20 = WebFeatureService(url=url, version='2.0.0')
    if wfs20:
        # Specify parameters (read data in json format) and fetch data from WFS using requests
        response = wfs20.getfeature(
            typename=type_name, bbox=box, srsname=srs, outputFormat='application/json')
        # Create GeoDataFrame from geojson and set coordinate reference system
        return gpd.GeoDataFrame.from_features(geojson.loads(response.read()), crs=srs)
    return None

def get_hydro_data(box, srs):
    surface = get_wfs_data("https://data.geopf.fr/wfs",
                            'BDTOPO_V3:surface_hydrographique',
                            box, srs)
    segment = get_wfs_data("https://data.geopf.fr/wfs",
                            'BDTOPO_V3:troncon_hydrographique',
                            box, srs)
    nodes = get_wfs_data("https://data.geopf.fr/wfs",
                         'BDTOPO_V3:noeud_hydrographique',
                            box, srs)
    if (surface is None) or (segment is None) or (nodes is None):
        print("No surface or no segment found.")
        return None
    print("Surfaces:", len(surface))
    print("Segments:", len(segment))
    print("Nodes:", len(nodes))
    surface = surface.query('nature == "Ecoulement naturel"')
    print("mask:", len(surface))
    surface = surface.query('persistance == "Permanent"')
    print("mask:", len(surface))
    box_as_polygon = geometry.box(box[0],box[1],box[2],box[3])
    surface = surface[surface.covered_by(box_as_polygon)]
    print("mask:", len(surface))
    # segment = segment.query('nature == "Ecoulement naturel" OR nature == "Conduit buse"')
    segment = segment[(segment["nature"] == "Ecoulement naturel") | (segment["nature"] == "Conduit buse")]
    print("mask:", len(segment))
    segment = segment[segment['liens_vers_surface_hydrographique'].isin(surface["cleabs"]) | (segment["nature"] == "Conduit buse")]
    print("mask:", len(segment))
    return (surface, segment, nodes)
