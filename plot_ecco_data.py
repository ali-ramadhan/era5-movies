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

# from PIL import Image
# Image.MAX_IMAGE_PIXELS = 1000000000


data_dir = Path("/storage6/alir/ecco4v4")
frames_dir = Path("frames")
colorbars_dir = Path("colorbars")
animations_dir = Path("animations")

for dir in [frames_dir, colorbars_dir, animations_dir]:
    if not dir.exists():
        dir.mkdir()

def plot_sst_frame(n, date):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 1000000000

    ds = xr.open_dataset(Path(data_dir, f"OCEAN_TEMPERATURE_SALINITY_day_mean_{date.year}-{date.month:02d}-{date.day:02d}_ECCO_V4r4_latlon_0p50deg.nc"))

    lat = ds.latitude
    lon = ds.longitude
    theta = ds.THETA

    filepath = Path(frames_dir, f"sst{n:05d}.png")
    print(f"Plotting {filepath}...")

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.pcolormesh(lon, lat, theta.isel(time=0, Z=0), **cm_tmp(units="K", levels=None, vmin=-2, vmax=32).cmap_kwargs)

    fname = Path("HYP_HR_SR", "HYP_HR_SR.tif")
    ax.imshow(mimage.imread(fname), origin="upper", transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90])

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, alpha=1, facecolor="None", edgecolor="black")
    # ax.add_feature(cfeature.LAKES.with_scale("50m"), linewidth=0.5, alpha=1, facecolor="None", edgecolor="black")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, alpha=1, facecolor="None", edgecolor="black")
    ax.axis("off")

    plt.tight_layout(pad=0)
    plt.savefig(filepath, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# def plot_temperature_colorbar():
#     fig, ax = plt.subplots(figsize=(6, 0.5))

#     cb = mpl.colorbar.ColorbarBase(ax, **cm_tmp(units="C", levels=None).cmap_kwargs, extend="both", orientation="horizontal")
#     cb.set_label("Temperature (°C)")

#     filepath = Path(colorbars_dir, "temperature_colorbar.png")
#     plt.savefig(filepath, transparent=True, dpi=300, bbox_inches="tight", pad_inches=0)

if "sst" in sys.argv:
    # plot_temperature_colorbar()

    dates = [datetime(2017, 1, 1) + timedelta(days=n) for n in range(365)]

    Parallel(n_jobs=min(24, os.cpu_count()))(delayed(plot_sst_frame)(n, d) for n, d in enumerate(dates))

    subprocess.run('ffmpeg -y -r 24 -f image2 -i frames/sst%05d.png -vcodec libx264 -preset veryslow -crf 25 -pix_fmt yuv420p -vf scale=iw/2:ih/2 animations/sst_h264_veryslow_crf25.mp4'.split())
