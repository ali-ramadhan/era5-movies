import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean

from paint.standard2 import cm_tmp, cm_pcp
from joblib import Parallel, delayed


data_dir = Path("/storage6/alir/oscar")
frames_dir = Path("frames")
colorbars_dir = Path("colorbars")
animations_dir = Path("animations")

for dir in [frames_dir, colorbars_dir, animations_dir]:
    if not dir.exists():
        dir.mkdir()

def plot_ocean_speed_frame(n, date):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 1000000000

    ds = xr.open_dataset(Path(data_dir, f"oscar_currents_final_{date.year}{date.month:02d}{date.day:02d}.nc4"))

    def magnitude(u, v):
        func = lambda x, y: np.sqrt(x**2 + y**2)
        return xr.apply_ufunc(func, u, v)

    speed = magnitude(ds.u, ds.v)

    lat = speed.lat
    lon = speed.lon

    filepath = Path(frames_dir, f"ocean_speed{n:05d}.png")
    print(f"Plotting {filepath}...")

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.pcolormesh(lon, lat, speed.isel(time=0).transpose(), cmap="Blues_r", vmin=0, vmax=2)

    fname = Path("HYP_HR_SR", "HYP_HR_SR_transparent.png")
    ax.imshow(mimage.imread(fname), origin="upper", transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90], zorder=10)

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, alpha=1, facecolor="None", edgecolor="black", zorder=11)
    # ax.add_feature(cfeature.LAKES.with_scale("50m"), linewidth=0.5, alpha=1, facecolor="None", edgecolor="black", zorder=11)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, alpha=1, facecolor="None", edgecolor="black", zorder=11)
    ax.axis("off")

    plt.tight_layout(pad=0)
    plt.savefig(filepath, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def plot_ocean_speed_colorbar():
    fig, ax = plt.subplots(figsize=(6, 0.5))

    cb = mpl.colorbar.ColorbarBase(ax, cmap="Blues_r", norm=mcolors.Normalize(0, 2), extend="max", orientation="horizontal")
    cb.set_label("Ocean current speed (m/s)", color='white')
    cb.ax.tick_params(colors='white')
    cb.outline.set_edgecolor('white')

    filepath = Path(colorbars_dir, "ocean_speed_colorbar.png")
    plt.savefig(filepath, transparent=True, dpi=300, bbox_inches="tight", pad_inches=0)

if "speed" in sys.argv:
    plot_ocean_speed_colorbar()

    dates = [datetime(2020, 1, 1) + timedelta(days=n) for n in range(366)]

    # Jobs seem to get stuck when using 16+ processes so I'm leaving it at 8 =/
    Parallel(n_jobs=min(8, os.cpu_count()))(delayed(plot_ocean_speed_frame)(n, d) for n, d in enumerate(dates))

    subprocess.run('ffmpeg -y -r 24 -f image2 -i frames/ocean_speed%05d.png -vcodec libx264 -preset veryslow -crf 25 -pix_fmt yuv420p -vf scale=iw/2:ih/2 animations/ocean_speed_h264_veryslow_crf25.mp4'.split())
