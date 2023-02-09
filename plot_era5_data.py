import sys
import os
from pathlib import Path

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmocean
import ffmpeg

from paint.standard2 import cm_tmp, cm_pcp
from joblib import Parallel, delayed

def plot_temperature_frame(n):
    ds = xr.open_dataset("2m_temperature.nc")

    t = ds.time
    lat = ds.latitude
    lon = ds.longitude
    T = ds.t2m

    filename = Path("frames", f"temperature{n:05d}.png")
    print(f"Plotting {filename}...")

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.pcolormesh(lon, lat, T.isel(time=n), **cm_tmp(units="K", levels=None).cmap_kwargs)
    ax.coastlines(resolution="50m", linewidth=0.5, alpha=1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)

def plot_temperature_colorbar():
    fig, ax = plt.subplots(figsize=(6, 0.5))

    cb = mpl.colorbar.ColorbarBase(ax, **cm_tmp(units="C", levels=None).cmap_kwargs, extend="both", orientation="horizontal")
    cb.set_label("Temperature (°C)")

    plt.savefig("temperature_colorbar.png", transparent=True, dpi=300, bbox_inches="tight", pad_inches=0)

if "temperature" in sys.argv:
    plot_temperature_colorbar()

    ds = xr.open_dataset("2m_temperature.nc")
    t = ds.time
    ds.close()

    Parallel(n_jobs=min(24, os.cpu_count()))(delayed(plot_temperature_frame)(n) for n in range(len(t)))

    (
        ffmpeg
        .input("frames/temperature%05d.png", framerate=24)
        .output("temperature.mp4")
        .overwrite_output()
        .run()
    )

    # png_filenames = [filename for filename in os.listdir(os.getcwd()) if filename.endswith(".png")]
    # print(f"Deleting {len(png_filenames)} leftover png files...")
    # [os.remove(filename) for filename in png_filenames]

def plot_precip_frame(n):
    ds = xr.open_dataset("temperature_and_total_precip.nc")

    t = ds.time
    lat = ds.latitude
    lon = ds.longitude
    P = ds.tp

    filename = Path("frames", f"precipitation{n:05d}.png")
    print(f"Plotting {filename}...")

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.pcolormesh(lon, lat, 1000 * P.isel(time=n), **cm_pcp(units="mm").cmap_kwargs)
    ax.coastlines(resolution="50m", linewidth=1, alpha=0.5)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)

def plot_precip_colorbar():
    fig, ax = plt.subplots(figsize=(6, 0.5))

    colors = np.array([
        "#ffffff",
        "#c7e9c0",
        "#a1d99b",
        "#74c476",
        "#31a353",
        "#006d2c",
        "#fffa8a",
        "#ffcc4f",
        "#fe8d3c",
        "#fc4e2a",
        "#d61a1c",
        "#ad0026",
        "#700026",
        "#3b0030",
        "#4c0073",
        "#ffdbff"
    ])

    bounds = (2 / 30) * np.array(
        [0, 0.01, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20, 30]
    )

    cmap = mcolors.LinearSegmentedColormap.from_list("precipitation", colors, N=len(colors))
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend="max")

    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.set_label("Precipitation (mm/hour)")

    plt.savefig("precipitation_colorbar.png", transparent=True, dpi=300, bbox_inches="tight", pad_inches=0)

if "precipitation" in sys.argv:
    plot_precip_colorbar()

    ds = xr.open_dataset("total_precipitation.nc")
    t = ds.time
    ds.close()

    Parallel(n_jobs=min(24, os.cpu_count()))(delayed(plot_precip_frame)(n) for n in range(len(t)))

    (
        ffmpeg
        .input("frames/precipitation%05d.png", framerate=24)
        .output("precipitation.mp4")
        .overwrite_output()
        .run()
)
