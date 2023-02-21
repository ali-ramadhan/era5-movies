import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean

from paint.standard2 import cm_tmp, cm_pcp
from joblib import Parallel, delayed


data_dir = Path("data")
frames_dir = Path("frames")
colorbars_dir = Path("colorbars")
animations_dir = Path("animations")

for dir in [frames_dir, colorbars_dir, animations_dir]:
    if not dir.exists():
        dir.mkdir()


def plot_temperature_frame(n):
    ds = xr.open_dataset(Path(data_dir, "2m_temperature_2018_12.nc"))

    t = ds.time
    lat = ds.latitude
    lon = ds.longitude
    T = ds.t2m

    filepath = Path(frames_dir, f"temperature{n:05d}.png")
    print(f"Plotting {filepath}...")

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.pcolormesh(lon, lat, T.isel(time=n), **cm_tmp(units="K", levels=None).cmap_kwargs)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, alpha=1, facecolor="None", edgecolor="black")
    ax.add_feature(cfeature.LAKES.with_scale("50m"), linewidth=0.5, alpha=1, facecolor="None", edgecolor="black")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, alpha=1, facecolor="None", edgecolor="black")
    ax.axis("off")

    plt.tight_layout(pad=0)
    plt.savefig(filepath, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def plot_temperature_colorbar():
    fig, ax = plt.subplots(figsize=(6, 0.5))

    cb = mpl.colorbar.ColorbarBase(ax, **cm_tmp(units="C", levels=None).cmap_kwargs, extend="both", orientation="horizontal")
    cb.set_label("Temperature (°C)")

    filepath = Path(colorbars_dir, "temperature_colorbar.png")
    plt.savefig(filepath, transparent=True, dpi=300, bbox_inches="tight", pad_inches=0)

if "temperature" in sys.argv:
    plot_temperature_colorbar()

    ds = xr.open_dataset(Path(data_dir, "2m_temperature_2018_12.nc"))
    t = ds.time
    ds.close()

    Parallel(n_jobs=min(24, os.cpu_count()))(delayed(plot_temperature_frame)(n) for n in range(len(t)))

    subprocess.run('ffmpeg -y -r 24 -f image2 -i frames/temperature%05d.png -vcodec libx264 -preset veryslow -crf 25 -pix_fmt yuv420p -vf scale=iw/2:ih/2 animations/temperature_h264_veryslow_crf25.mp4'.split())


def precipitation_cmap():
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

    max = 10
    bounds = (max / 30) * np.array(
        [0, 0.01, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20, 30]
    )

    cmap = mcolors.LinearSegmentedColormap.from_list("precipitation", colors, N=len(colors))
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend="max")

    return cmap, norm

def plot_precipitation_frame(n):
    ds = xr.open_dataset(Path(data_dir, "total_precipitation_2017_08_16-31.nc"))

    t = ds.time
    lat = ds.latitude
    lon = ds.longitude
    P = ds.tp

    filepath = Path(frames_dir, f"precipitation{n:05d}.png")
    print(f"Plotting {filepath}...")

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    cmap, norm = precipitation_cmap()
    ax.pcolormesh(lon, lat, 1000 * P.isel(time=n), cmap=cmap, norm=norm)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, alpha=1, facecolor="None", edgecolor="black")
    ax.add_feature(cfeature.LAKES.with_scale("50m"), linewidth=0.5, alpha=1, facecolor="None", edgecolor="black")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, alpha=1, facecolor="None", edgecolor="black")
    ax.axis("off")

    plt.tight_layout(pad=0)
    plt.savefig(filepath, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def plot_precipitation_colorbar():
    fig, ax = plt.subplots(figsize=(6, 0.5))

    cmap, norm = precipitation_cmap()
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.set_label("Precipitation (mm/hour)")

    filepath = Path(colorbars_dir, "precipitation_colorbar.png")
    plt.savefig(filepath, transparent=True, dpi=300, bbox_inches="tight", pad_inches=0)

if "precipitation" in sys.argv:
    plot_precipitation_colorbar()

    ds = xr.open_dataset(Path(data_dir, "total_precipitation_2017_08_16-31.nc"))
    t = ds.time
    ds.close()

    Parallel(n_jobs=min(24, os.cpu_count()))(delayed(plot_precipitation_frame)(n) for n in range(len(t)))

    subprocess.run('ffmpeg -y -r 24 -f image2 -i frames/precipitation%05d.png -vcodec libx264 -preset veryslow -crf 25 -pix_fmt yuv420p -vf scale=iw/2:ih/2 animations/precipitation_h264_veryslow_crf25.mp4'.split())
