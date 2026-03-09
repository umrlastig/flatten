import time
import requests
import logging

logger = logging.getLogger(__name__)

API_URL = "https://data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevation.json?lon={lon}&lat={lat}&resource=ign_rge_alti_wld&delimiter=|&indent=false&measures=false&zonly=true"


def send_points(points):
    """Send points to the API."""
    p_lat, p_lon = zip(*points)
    response = requests.get(
        API_URL.format(lon="|".join(map(str, p_lon)), lat="|".join(map(str, p_lat)))
    )
    response.raise_for_status()  # raise on HTTP error
    return response.json()


def throttle_requests(points, max_rps=5):
    """
    Send `points` to the API without exceeding `max_rps` requests per second.
    """
    batch_size = max_rps  # 5 requests per second for gpf according to the doc
    res = []
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        start = time.time()
        try:
            result = send_points(batch)
            res.extend(result["elevations"])
            logger.info("Sucess", result)
        except Exception as exc:
            logger.error("Error", exc)
            res.extend([-1000] * len(batch))
        # Sleep the remainder of the second (if any)
        elapsed = time.time() - start
        sleep_time = max(0, 1.0 - elapsed)
        if sleep_time:
            time.sleep(sleep_time)
    return res
