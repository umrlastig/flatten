import pytest

from flatten.wfs import get_hydro_data, get_wfs_data

def test_get_wfs_data():
    srs = "urn:ogc:def:crs:EPSG::2154"
    in_box = (1038460, 6295170, 1038520, 6295240)
    box = (in_box[0], in_box[1], in_box[2], in_box[3], srs)
    surface = get_wfs_data(
        "https://data.geopf.fr/wfs", "BDTOPO_V3:surface_hydrographique", box, srs
    )
    print(surface)
    assert surface is not None
    if surface is not None:
        assert len(surface) == 2
        cleabs = ["SURF_EAU0000000073987688","SURF_EAU0000002009210739"]
        print(sorted(surface["cleabs"].tolist()))
        assert sorted(surface["cleabs"].tolist()) == cleabs

def test_get_hydro_data():
    srs = "urn:ogc:def:crs:EPSG::2154"
    in_box = (1038460, 6295170, 1038520, 6295240)
    box = (in_box[0], in_box[1], in_box[2], in_box[3], srs)
    (surfaces, segments, nodes) = get_hydro_data(box, srs)
    print(surfaces)
    assert surfaces is not None
    assert len(surfaces) == 0
    assert segments is None
    assert nodes is None