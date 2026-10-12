"""Kickoff weather forecasts for upcoming outdoor games (Open-Meteo, free, no API key).

nflverse only records temperature and wind after a game is played, so without this every
upcoming game looks calm. Forecasts reach about 16 days ahead and are refreshed every run,
so they sharpen as kickoff gets closer. If the service cannot be reached, games keep the
league-typical values (no wind flags) and the run continues.
"""
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

from geo import VENUES

API = "https://api.open-meteo.com/v1/forecast"
HORIZON_DAYS = 15


def _kickoff_utc(gameday, gametime) -> pd.Timestamp:
    t = gametime if isinstance(gametime, str) and ":" in gametime else "13:00"
    return pd.Timestamp(f"{str(gameday)[:10]} {t}").tz_localize("America/New_York").tz_convert("UTC")


def _forecast(lat: float, lon: float, start, end) -> pd.DataFrame:
    r = requests.get(API, timeout=20, params={
        "latitude": lat, "longitude": lon, "hourly": "temperature_2m,wind_speed_10m",
        "wind_speed_unit": "mph", "temperature_unit": "fahrenheit", "timezone": "UTC",
        "start_date": start.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d")})
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame({"time": pd.to_datetime(h["time"]).tz_localize("UTC"),
                         "temp": h["temperature_2m"], "wind": h["wind_speed_10m"]})


def fill_forecasts(g: pd.DataFrame) -> pd.DataFrame:
    """Fill temp and wind for unplayed outdoor games within the forecast window.
    Averages the three hours starting at kickoff."""
    g = g.copy()
    g["weather_source"] = np.where(g.wind.notna(), "observed", "none")
    now = pd.Timestamp.now(tz="UTC")
    todo = g[g.result.isna() & g.wind.isna() & (g.roof == "outdoors")]
    filled = 0
    for idx, row in todo.iterrows():
        ko = _kickoff_utc(row.gameday, row.gametime)
        if not (now - timedelta(hours=4) <= ko <= now + timedelta(days=HORIZON_DAYS)):
            continue
        venue = VENUES.get(str(row.stadium_id)[:3])
        if venue is None:
            continue
        try:
            fc = _forecast(venue[0], venue[1], ko, ko + timedelta(hours=3))
        except Exception as e:  # never let weather break the run
            print("weather forecast unavailable:", type(e).__name__)
            break
        win = fc[(fc.time >= ko.floor("h")) & (fc.time <= ko.floor("h") + timedelta(hours=2))]
        if win.empty:
            continue
        g.loc[idx, ["temp", "wind"]] = [round(win.temp.mean(), 1), round(win.wind.mean(), 1)]
        g.loc[idx, "weather_source"] = "forecast"
        filled += 1
    if len(todo):
        print(f"weather: forecast filled for {filled} of {len(todo)} upcoming outdoor games")
    return g
