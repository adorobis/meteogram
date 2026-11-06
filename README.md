# meteo.pl meteogram

Draws a meteo.pl-style meteogram (temperature and precipitation panels) from the
ICM **api.meteo.pl** WRF forecast.

```bash
pip install -r requirements.txt
export METEO_API_KEY=...        # https://api.meteo.pl/user/settings/
python meteogram.py --lat 50.11 --lon 18.40 --name Gaszowice -o gaszowice.png
```

Options: `--run 2026-08-08T00` (default: latest run), `--start 2026-08-09T14`
(local time; default: now, or the run start if the run is older), `--hours 60`,
`--grid d01_XLONG_XLAT` (coarser grid), `--rowcol 284,386` (skip the lat/lon lookup).

The free test point is Białowieża (the default location, grid point 284,386 on
`wrf/d02_XLONG_XLAT`). Other points are charged to your api.meteo.pl balance.
Each new image costs 7 forecast requests (one per WRF field: T2, Q2, U10, V10, RAINNC,
RAINC, SNOWNC). api.meteo.pl charges 0.01 PLN per field per location per model run, so a
meteogram for a paid location costs 0.07 PLN. Top-ups start at 5 PLN.

## Local cache

Downloaded data is saved under `~/.cache/meteogram` (or `$XDG_CACHE_HOME/meteogram`).
A published model run never changes, so drawing the same run and location again
(another time window, another place name, a restyled chart) reads from disk and makes
no paid requests. The program prints how many paid requests it made.

* grid-point lookups are kept permanently
* the list of available runs is reused for 1 hour, so new runs are still found
* `--refresh` re-downloads and overwrites, `--no-cache` turns the cache off,
  `--cache-dir DIR` puts it somewhere else
* delete the directory to clear it

What is drawn:
| element | source |
|---|---|
| red line | 2 m temperature (T2) |
| blue line + fill | perceived temp (wind chill, or heat index at 27 °C or above) |
| purple dots | dew point computed from Q2 at standard pressure (1013 hPa) |
| green bars | rain, mm/h: hourly increase of RAINNC + RAINC, minus snow |
| blue bars | snow, mm/h water equivalent: hourly increase of SNOWNC (convective precipitation counts as rain) |
| grey bands | night (calculated sunrise/sunset) |

Times are shown in Europe/Warsaw local time.
