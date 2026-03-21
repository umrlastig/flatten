import pytest

from flatten.elevation import send_points, throttle_requests

def test_send_points():
    points = [(48.853088, 2.348968)]
    result = send_points(points)
    print(result)
    assert result["elevations"][0] == 34.54

def test_throttle_requests():
    points = [(48.853088, 2.348968),(51.4773284,-0.0007195)]
    result = throttle_requests(points)
    print(result)
    assert len(result) == 2
    assert result[0] == 34.54
    assert result[1] == -99999.0
    points = [(200.0,200.0)]
    result = throttle_requests(points)
    print(result)
    assert len(result) == 1
    assert result[0] == -1000
