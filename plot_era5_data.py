import os
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cmocean
import ffmpeg

from joblib import Parallel, delayed

def plot_frame(n):
    ds = xr.open_dataset("temperature_and_total_precip.nc")

    t = ds.time
    lat = ds.latitude
    lon = ds.longitude
    T = ds.t2m
    P = ds.tp

    filename = f"precip{n:05d}.png"
    print(f"Plotting {filename}...")
    
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.pcolormesh(lon, lat, P.isel(time=n), cmap=cmocean.cm.rain, vmin=0, vmax=2.5e-3)
    ax.coastlines(resolution="50m", linewidth=1, alpha=0.5)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)

ds = xr.open_dataset("temperature_and_total_precip.nc")
t = ds.time
ds.close()

Parallel(n_jobs=os.cpu_count())(delayed(plot_frame)(n) for n in range(len(t)))

(
    ffmpeg
    .input("precip%05d.png", framerate=15)
    .output("precip.webm")
    .overwrite_output()
    .run()
)

png_filenames = [filename for filename in os.listdir(os.getcwd()) if filename.endswith(".png")]
print(f"Deleting {len(png_filenames)} leftover png files...")
[os.remove(filename) for filename in png_filenames]
