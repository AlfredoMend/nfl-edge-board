"""Stadium locations for travel distance and time-zone shifts."""
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from zoneinfo import ZoneInfo

# nflverse stadium_id prefix -> (lat, lon, IANA time zone)
VENUES = {
    "ATL": (33.755, -84.401, "America/New_York"), "BAL": (39.278, -76.623, "America/New_York"),
    "BOS": (42.091, -71.264, "America/New_York"), "BUF": (42.774, -78.787, "America/New_York"),
    "CAR": (35.226, -80.853, "America/New_York"), "CHI": (41.862, -87.617, "America/Chicago"),
    "CIN": (39.095, -84.516, "America/New_York"), "CLE": (41.506, -81.700, "America/New_York"),
    "DAL": (32.748, -97.093, "America/Chicago"), "DEN": (39.744, -105.020, "America/Denver"),
    "DET": (42.340, -83.046, "America/Detroit"), "GNB": (44.501, -88.062, "America/Chicago"),
    "HOU": (29.685, -95.411, "America/Chicago"), "IND": (39.760, -86.164, "America/Indiana/Indianapolis"),
    "JAX": (30.324, -81.637, "America/New_York"), "KAN": (39.049, -94.484, "America/Chicago"),
    "LAX": (33.953, -118.339, "America/Los_Angeles"), "MIA": (25.958, -80.239, "America/New_York"),
    "MIN": (44.974, -93.258, "America/Chicago"), "NAS": (36.166, -86.771, "America/Chicago"),
    "NOR": (29.951, -90.081, "America/Chicago"), "NYC": (40.814, -74.074, "America/New_York"),
    "OAK": (37.752, -122.201, "America/Los_Angeles"), "PHI": (39.901, -75.168, "America/New_York"),
    "PHO": (33.528, -112.263, "America/Phoenix"), "PIT": (40.447, -80.016, "America/New_York"),
    "SDG": (32.783, -117.120, "America/Los_Angeles"), "SEA": (47.595, -122.332, "America/Los_Angeles"),
    "SFO": (37.403, -121.970, "America/Los_Angeles"), "STL": (38.633, -90.189, "America/Chicago"),
    "TAM": (27.976, -82.503, "America/New_York"), "VEG": (36.091, -115.184, "America/Los_Angeles"),
    "WAS": (38.908, -76.864, "America/New_York"),
    # international and neutral sites
    "LON": (51.556, -0.280, "Europe/London"), "MEX": (19.303, -99.150, "America/Mexico_City"),
    "GER": (48.219, 11.625, "Europe/Berlin"), "MUN": (48.219, 11.625, "Europe/Berlin"),
    "FRA": (50.069, 8.645, "Europe/Berlin"), "SAO": (-23.545, -46.474, "America/Sao_Paulo"),
    "RIO": (-22.912, -43.230, "America/Sao_Paulo"), "MAD": (40.453, -3.688, "Europe/Madrid"),
    "PAR": (48.924, 2.360, "Europe/Paris"), "MEL": (-37.820, 144.983, "Australia/Melbourne"),
    "DUB": (53.335, -6.228, "Europe/Dublin"), "TOR": (43.641, -79.389, "America/Toronto"),
}

TEAM_HOME = {
    "ARI": "PHO", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF", "CAR": "CAR", "CHI": "CHI", "CIN": "CIN",
    "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GB": "GNB", "HOU": "HOU", "IND": "IND",
    "JAX": "JAX", "KC": "KAN", "LA": "LAX", "LAC": "LAX", "LV": "VEG", "MIA": "MIA", "MIN": "MIN",
    "NE": "BOS", "NO": "NOR", "NYG": "NYC", "NYJ": "NYC", "OAK": "OAK", "PHI": "PHI", "PIT": "PIT",
    "SD": "SDG", "SEA": "SEA", "SF": "SFO", "STL": "STL", "TB": "TAM", "TEN": "NAS", "WAS": "WAS",
}


def miles(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (a[0], a[1], b[0], b[1]))
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * asin(sqrt(h))


def utc_offset(tz: str, when: datetime) -> float:
    return ZoneInfo(tz).utcoffset(when).total_seconds() / 3600


def travel(team: str, venue_code: str, when: datetime) -> tuple[float, float]:
    """Miles from the team's home stadium to the venue, and hours of clock shift (venue - home)."""
    home = VENUES.get(TEAM_HOME.get(team, ""), None)
    venue = VENUES.get(venue_code, home)
    if home is None or venue is None:
        return 0.0, 0.0
    return miles(home, venue), utc_offset(venue[2], when) - utc_offset(home[2], when)
