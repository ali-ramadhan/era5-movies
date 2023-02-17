import sys
import cdsapi
from pathlib import Path

days = [
    "01", "02", "03",
    "04", "05", "06",
    "07", "08", "09",
    "10", "11", "12",
    "13", "14", "15",
    "16", "17", "18",
    "19", "20", "21",
    "22", "23", "24",
    "25", "26", "27",
    "28", "29", "30",
    "31"
]

times = [
    "00:00", "01:00", "02:00",
    "03:00", "04:00", "05:00",
    "06:00", "07:00", "08:00",
    "09:00", "10:00", "11:00",
    "12:00", "13:00", "14:00",
    "15:00", "16:00", "17:00",
    "18:00", "19:00", "20:00",
    "21:00", "22:00", "23:00"
]

data_dir = Path("data")
if not data_dir.exists():
    data_dir.mkdir()

c = cdsapi.Client()

if "temperature" in sys.argv:
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": "2m_temperature",
            "year": "2018",
            "month": "12",
            "day": days,
            "time": times,
        },
        Path(data_dir, "2m_temperature_2018_12.nc")
    )

if "precipitation" in sys.argv:
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": "total_precipitation",
            "year": "2017",
            "month": "08",
            "day": [
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "time": times,
        },
        Path(data_dir, "total_precipitation_2017_08_16-31.nc")
    )
