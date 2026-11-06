#!/usr/bin/env python3
"""Meteogram generator for api.meteo.pl (ICM, University of Warsaw).

Fetches a WRF (or other) point forecast from https://api.meteo.pl and renders
a two-panel PNG in the style of the classic meteo.pl meteogram:

  * top panel   - air temperature (red), perceived temperature (blue),
                  dew point (purple dots)
  * bottom panel- precipitation in mm/h: rain (green bars), snow (blue bars)

Night hours are shaded grey, the header shows sunrise/sunset, the place name,
coordinates and model.

Usage:
    export METEO_API_KEY=...            # key from https://api.meteo.pl/user/settings/
    python meteogram.py --lat 52.701 --lon 23.841 --name "Białowieża" -o out.png
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402

API_ROOT = "https://api.meteo.pl/api/v1"
TZ = ZoneInfo("Europe/Warsaw")
PL_DAYS = ["Pon", "Wt", "Śr", "Czw", "Pt", "Sob", "Ndz"]

# WRF surface fields used by the meteogram (all on level "0").
WRF_FIELDS = ["T2", "Q2", "U10", "V10", "RAINNC", "RAINC", "SNOWNC"]
STD_PRESSURE_PA = 101325.0          # used for dew point instead of the (paid) PSFC field
DEFAULT_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "meteogram"
DATES_TTL_S = 3600                  # how long the list of available runs is reused


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
class MeteoApi:
    """Thin client with an on-disk cache.

    A forecast for a given (model, grid, point, field, level, run) never changes once
    published, so it is stored forever and served from disk afterwards (no charge).
    Grid-point lookups are cached forever too; the list of available runs is cached
    for DATES_TTL_S so new runs are still picked up.
    """

    def __init__(self, api_key: str | None, model: str, grid: str, timeout: int = 60,
                 cache_dir: Path | None = DEFAULT_CACHE, refresh: bool = False):
        self.model, self.grid, self.timeout = model, grid, timeout
        self.cache_dir, self.refresh = cache_dir, refresh
        self.api_calls = self.cache_hits = 0
        self.s = requests.Session()
        if api_key:
            self.s.headers["Authorization"] = f"Token {api_key}"

    @property
    def base(self) -> str:
        return f"{API_ROOT}/model/{self.model}/grid/{self.grid}"

    def _get(self, url: str):
        r = self.s.get(url, timeout=self.timeout)
        self._check(r)
        return r.json()

    # --- cache helpers -------------------------------------------------------
    def _cpath(self, *parts: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir.joinpath(self.model, self.grid, *parts)

    def _cread(self, path: Path | None, max_age: float | None = None):
        if path is None or self.refresh or not path.exists():
            return None
        if max_age is not None and time.time() - path.stat().st_mtime > max_age:
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None                                   # corrupt file -> refetch
        self.cache_hits += 1
        return data

    @staticmethod
    def _cwrite(path: Path | None, data) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)                                 # atomic: no half-written files

    @staticmethod
    def _check(r: requests.Response) -> None:
        if not r.ok:
            raise RuntimeError(f"{r.request.method} {r.url} -> {r.status_code}: {r.text[:300]}")

    def latlon_to_rowcol(self, lat: float, lon: float) -> tuple[int, int]:
        path = self._cpath("rowcol", f"{lat:.5f},{lon:.5f}.json")
        pt = self._cread(path)
        if pt is None:
            pt = self._get(f"{self.base}/latlon2rowcol/{lat},{lon}/")["points"][0]
            self._cwrite(path, pt)
        return pt["row"], pt["col"]

    def available_dates(self, row: int, col: int, field: str, level: str) -> list[dt.datetime]:
        """Expand the compressed run list ({starting-date, interval[h], count})."""
        url = f"{self.base}/coordinates/{row},{col}/field/{field}/level/{level}/date/"
        path = self._cpath("dates", f"{row},{col}", f"{field}_{level}.json")
        dates = self._cread(path, max_age=DATES_TTL_S)
        if dates is None:
            dates = self._get(url)["dates"]
            self._cwrite(path, dates)
        out = []
        for d in dates:
            start = dt.datetime.strptime(d["starting-date"], "%Y-%m-%dT%H").replace(tzinfo=dt.timezone.utc)
            out += [start + dt.timedelta(hours=d["interval"] * i) for i in range(d["count"])]
        return sorted(out)

    def forecast(self, row: int, col: int, field: str, level: str, run: dt.datetime):
        url = (f"{self.base}/coordinates/{row},{col}/field/{field}/level/{level}"
               f"/date/{run:%Y-%m-%dT%H}/forecast/")
        path = self._cpath("forecast", f"{row},{col}", f"{run:%Y-%m-%dT%H}", f"{field}_{level}.json")
        js = self._cread(path)
        if js is None:
            js = self._fetch_forecast(url)
            if js.get("times"):                           # never cache an empty answer
                self._cwrite(path, js)
        times = np.array([dt.datetime.fromisoformat(t.replace("Z", "+00:00")) for t in js["times"]])
        return times, np.array(js["data"], dtype=float)

    def _fetch_forecast(self, url: str) -> dict:
        self.api_calls += 1
        for attempt in range(5):                          # server throttles bursts with 503
            r = self.s.post(url, timeout=self.timeout)
            if r.status_code not in (429, 502, 503, 504):
                break
            time.sleep(2 ** attempt)
        self._check(r)
        return r.json()


# --------------------------------------------------------------------------- #
# Meteorology helpers
# --------------------------------------------------------------------------- #
def dew_point_c(q2: np.ndarray, psfc_pa: np.ndarray) -> np.ndarray:
    """Dew point from specific humidity / mixing ratio (kg/kg) and pressure (Pa)."""
    e_hpa = q2 * psfc_pa / (0.622 + 0.378 * q2) / 100.0
    e_hpa = np.clip(e_hpa, 1e-3, None)
    g = np.log(e_hpa / 6.112)
    return 243.5 * g / (17.67 - g)


def perceived_c(t_c: np.ndarray, wind_ms: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """Wind chill for cold/windy, heat index (Steadman/Rothfusz) for hot, else T."""
    v = wind_ms * 3.6
    wc = 13.12 + 0.6215 * t_c - 11.37 * v ** 0.16 + 0.3965 * t_c * v ** 0.16
    out = np.where((t_c <= 10) & (v > 4.8), wc, t_c)
    # relative humidity from dew point
    rh = 100 * np.exp(17.625 * td_c / (243.04 + td_c)) / np.exp(17.625 * t_c / (243.04 + t_c))
    tf = t_c * 9 / 5 + 32
    hi = (-42.379 + 2.04901523 * tf + 10.14333127 * rh - 0.22475541 * tf * rh
          - 6.83783e-3 * tf ** 2 - 5.481717e-2 * rh ** 2 + 1.22874e-3 * tf ** 2 * rh
          + 8.5282e-4 * tf * rh ** 2 - 1.99e-6 * tf ** 2 * rh ** 2)
    return np.where(t_c >= 27, (hi - 32) * 5 / 9, out)


def sun_times(day: dt.date, lat: float, lon: float):
    """Sunrise/sunset (UTC datetimes) via the NOAA solar algorithm. None if polar."""
    n = day.timetuple().tm_yday
    gamma = 2 * math.pi / 365 * (n - 1)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    la = math.radians(lat)
    cos_ha = (math.cos(math.radians(90.833)) / (math.cos(la) * math.cos(decl))
              - math.tan(la) * math.tan(decl))
    if abs(cos_ha) > 1:
        return None
    ha = math.degrees(math.acos(cos_ha))
    midnight = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
    rise = midnight + dt.timedelta(minutes=720 - 4 * (lon + ha) - eqtime)
    sset = midnight + dt.timedelta(minutes=720 - 4 * (lon - ha) - eqtime)
    return rise, sset


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
@dataclass
class Meteo:
    times: np.ndarray       # tz-aware local datetimes
    temp: np.ndarray        # °C
    dew: np.ndarray         # °C
    feel: np.ndarray        # °C
    rain: np.ndarray        # mm/h (liquid)
    snow: np.ndarray        # mm/h water equivalent
    run: dt.datetime


def load(api: MeteoApi, row: int, col: int, run: dt.datetime | None) -> Meteo:
    if run is None:
        run = api.available_dates(row, col, "T2", "0")[-1]
    with ThreadPoolExecutor(max_workers=3) as ex:
        res = dict(zip(WRF_FIELDS, ex.map(lambda f: api.forecast(row, col, f, "0", run), WRF_FIELDS)))
    times = res["T2"][0]
    d = {k: v[1] for k, v in res.items()}
    t = d["T2"] - 273.15
    td = np.minimum(dew_point_c(d["Q2"], STD_PRESSURE_PA), t)
    wind = np.hypot(d["U10"], d["V10"])
    def hourly(acc):                                     # accumulated mm -> mm/h
        return np.clip(np.diff(acc, prepend=acc[0]), 0, None)
    total = hourly(d["RAINNC"] + d["RAINC"])             # all precip (liquid + frozen)
    # grid-scale snow (water equivalent) is accumulated in SNOWNC; convective
    # precipitation (RAINC) is always counted as rain
    snow = np.minimum(hourly(d["SNOWNC"]), total)
    rain = total - snow
    return Meteo(np.array([x.astimezone(TZ) for x in times]), t, td, perceived_c(t, wind, td),
                 rain, snow, run)


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
NAVY = "#1a237e"


def utc_offset_label(t: dt.datetime) -> str:
    h = int(t.utcoffset().total_seconds() // 3600)
    return f"UTC{h:+d}"


def render(m: Meteo, lat: float, lon: float, name: str, model_label: str,
           start: dt.datetime, hours: int, out: str) -> None:
    end = start + dt.timedelta(hours=hours)
    sel = (m.times >= start) & (m.times <= end)
    if not sel.any():
        raise SystemExit("No forecast data in the requested window.")
    tt, T, Td, F, Rn, Sn = (a[sel] for a in (m.times, m.temp, m.dew, m.feel, m.rain, m.snow))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(6.6, 2.8), dpi=150)
    ax1 = fig.add_axes([0.195, 0.43, 0.79, 0.30])
    ax2 = fig.add_axes([0.195, 0.03, 0.79, 0.33], sharex=ax1)

    # --- header ---------------------------------------------------------------
    st = sun_times(start.date(), lat, lon)
    if st:
        rise, sset = (x.astimezone(TZ) for x in st)
        off = utc_offset_label(rise)
        fig.text(0.135, 0.955, f"wschód słońca: {rise:%H:%M} {off}", color=NAVY, fontsize=10)
        fig.text(0.135, 0.895, f"zachód słońca: {sset:%H:%M} {off}", color=NAVY, fontsize=10)
    fig.text(0.985, 0.955, f"{name} ({lat:.2f}, {lon:.2f})", color=NAVY, ha="right", fontsize=10)
    fig.text(0.985, 0.895, f"{model_label}  run {m.run:%Y-%m-%d %HZ}", color=NAVY, ha="right", fontsize=9)

    # --- night shading ---------------------------------------------------------
    day = start.date() - dt.timedelta(days=1)
    while day <= end.date() + dt.timedelta(days=1):
        a, b = sun_times(day, lat, lon) or (None, None)
        nxt = sun_times(day + dt.timedelta(days=1), lat, lon)
        if b and nxt:
            for ax in (ax1, ax2):
                ax.axvspan(b.astimezone(TZ), nxt[0].astimezone(TZ), color="#e3e3e3", zorder=0, lw=0)
        day += dt.timedelta(days=1)

    # --- temperature panel ----------------------------------------------------
    lo = min(-5, math.floor(min(F.min(), Td.min()) / 5) * 5)
    hi = max(10, math.ceil(T.max() / 5) * 5)
    ax1.set_ylim(lo, hi)
    ax1.fill_between(tt, lo, F, color="#8ecae6", alpha=0.85, lw=0, zorder=1)
    ax1.fill_between(tt, F, T, color="#ffcdd2", alpha=0.8, lw=0, zorder=1)
    ax1.plot(tt, F, color="#1e88e5", lw=1.6, zorder=3)
    ax1.plot(tt, T, color="#e53935", lw=1.6, zorder=4)
    ax1.plot(tt, Td, ls="none", marker=".", ms=3, color="#7e57c2", zorder=4)
    step = 5 if hi - lo <= 20 else 10
    ax1.set_yticks([v for v in range(lo, hi + 1, 5) if v % step == 0])
    ax1.set_ylabel("temperatura\n(°C)", color=NAVY)

    # --- precipitation panel --------------------------------------------------
    ptop = max(5, math.ceil((Rn + Sn).max()))
    ax2.set_ylim(0, ptop)
    w = 0.75 / 24                                         # bar width in days (45 min)
    # value at time t is the precipitation of the preceding hour -> centre bar on t-30min
    bx = [x - dt.timedelta(minutes=30) for x in tt]
    ax2.bar(bx, Rn, width=w, color="#2e9e3e", zorder=3, lw=0)
    ax2.bar(bx, Sn, width=w, bottom=Rn, color="#1e6fd9", zorder=3, lw=0)
    ax2.set_yticks(range(0, ptop + 1, max(1, ptop // 5)))
    ax2.set_ylabel("opad\n(mm/h)", color=NAVY)

    # --- shared x axis / grid ------------------------------------------------
    x0 = start.replace(minute=0, second=0, microsecond=0)
    ax1.set_xlim(x0, end)
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(x0.hour % 3, 24, 3), tz=TZ))
        ax.grid(True, ls=(0, (1, 2)), color="black", lw=0.7, zorder=2)
        ax.set_axisbelow(False)
        for s in ax.spines.values():
            s.set_linewidth(0.8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H", tz=TZ))
    ax1.xaxis.tick_top()
    ax1.tick_params(axis="x", length=0, pad=2, labelsize=10)
    ax2.tick_params(axis="x", length=0, labelbottom=False)

    # date labels at local midnight
    d = (x0 + dt.timedelta(days=1)).replace(hour=0)
    while d < end - dt.timedelta(hours=8):
        ax1.annotate(f"{PL_DAYS[d.weekday()]}, {d:%d.%m}", (mdates.date2num(d), 1),
                     xycoords=("data", "axes fraction"), xytext=(0, 15),
                     textcoords="offset points", fontsize=9.5, ha="left")
        d += dt.timedelta(days=1)

    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Draw a meteo.pl-style meteogram from api.meteo.pl")
    p.add_argument("--lat", type=float, default=52.701, help="latitude (default: Białowieża free point)")
    p.add_argument("--lon", type=float, default=23.841, help="longitude")
    p.add_argument("--name", default="Białowieża", help="place name shown in the header")
    p.add_argument("--model", default="wrf")
    p.add_argument("--grid", default="d02_XLONG_XLAT")
    p.add_argument("--rowcol", help="grid point 'row,col' (skips lat/lon lookup)")
    p.add_argument("--run", help="model run, e.g. 2026-08-08T00 (default: latest)")
    p.add_argument("--start", help="local start time YYYY-mm-ddTHH (default: now, or run start if run is older)")
    p.add_argument("--hours", type=int, default=60, help="window length in hours (default 60)")
    p.add_argument("-o", "--output", default="meteogram.png")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                   help=f"local cache for API responses (default: {DEFAULT_CACHE})")
    p.add_argument("--no-cache", action="store_true", help="do not read or write the cache")
    p.add_argument("--refresh", action="store_true", help="ignore cached data, re-download and overwrite it")
    p.add_argument("--api-key", default=os.environ.get("METEO_API_KEY"),
                   help="API key (default: $METEO_API_KEY)")
    a = p.parse_args(argv)

    if a.model != "wrf":
        p.error("only the WRF model is currently supported (field names are model-specific)")

    api = MeteoApi(a.api_key, a.model, a.grid,
                   cache_dir=None if a.no_cache else a.cache_dir, refresh=a.refresh)
    if a.rowcol:
        row, col = map(int, a.rowcol.split(","))
    else:
        row, col = api.latlon_to_rowcol(a.lat, a.lon)
    run = (dt.datetime.strptime(a.run, "%Y-%m-%dT%H").replace(tzinfo=dt.timezone.utc)
           if a.run else None)

    m = load(api, row, col, run)
    if a.start:
        start = dt.datetime.strptime(a.start, "%Y-%m-%dT%H").replace(tzinfo=TZ)
    else:
        now = dt.datetime.now(TZ).replace(minute=0, second=0, microsecond=0)
        start = now if m.times[0] <= now <= m.times[-1] else m.times[0]

    grid_label = a.grid.split("_")[0]
    render(m, a.lat, a.lon, a.name, f"{a.model.upper()} {grid_label}", start, a.hours, a.output)
    print(f"grid point {row},{col}  run {m.run:%Y-%m-%dT%H}Z  -> {a.output}  "
          f"(paid forecast requests: {api.api_calls}, served from cache: {api.cache_hits})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
