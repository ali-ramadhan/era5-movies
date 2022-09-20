import numpy as np
import xarray as xr

ds = xr.open_dataset("temperature.nc")
ds.t2m.data.tofile("temperature.bin")
