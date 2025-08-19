import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
from parflow import Run
from parflow.tools.io import read_pfb, read_clm, write_pfb
from parflow.tools.fs import mkdir
from parflow.tools.settings import set_working_directory
import parflow.tools.hydrology as hydro
import subsettools as st
import hf_hydrodata as hf

# You need to register on https://hydrogen.princeton.edu/pin before you can use the hydrodata utilities
hf.register_api_pin("<your_email>", "<your_pin>")

my_runname = "my_conus2"

# provide a way to create a subset from the conus domain (huc, lat/lon bbox currently supported)
hucs = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18"]
# provide information about the datasets you want to access for run inputs using the data catalog
start = "2005-10-01"
wy = 2006
#end = "2005-10-03"
grid = "conus2"
var_ds = "conus2_domain"
#forcing_ds = "CW3E"
# cluster topology
P = 1
Q = 1

# set the directory paths where you want to write your subset files
home = os.path.expanduser("/home/lp9617/workspace/")
base_dir = os.path.join(home, "conus2_spinups/runs/")
input_dir = os.path.join(base_dir, "inputs/", f"{my_runname}_{grid}_{wy}_WY_spinup")
output_dir = os.path.join(base_dir, "outputs/")
static_write_dir = os.path.join(input_dir, "static")
mkdir(static_write_dir)
# pf_out_dir = os.path.join(output_dir, f"{my_runname}_{grid}_{wy}_WY_spinup")
# mkdir(pf_out_dir)

# Set the PARFLOW_DIR path to your local installation of ParFlow.
# This is only necessary if this environment variable is not already set.
os.environ["PARFLOW_DIR"] = "/home/lp9617/workspace/parflow"

# load your preferred template runscript
# reference_run = st.get_template_runscript(grid, "transient", "solid", pf_out_dir)

ij_bounds, mask = st.define_huc_domain(hucs=hucs, grid=grid)
print("ij_bound returns [imin, jmin, imax, jmax]")
print(f"bounding box: {ij_bounds}")

nj = ij_bounds[3] - ij_bounds[1]
ni = ij_bounds[2] - ij_bounds[0]
print(f"nj: {nj}")
print(f"ni: {ni}")

plt.figure()
plt.imshow(mask, origin='lower')
plt.colorbar()
plt.savefig('/home/lp9617/workspace/conus2_spinups/runs/inputs/mask.png')
plt.close()

mask_solid_paths = st.write_mask_solid(mask=mask, grid=grid, write_dir=static_write_dir)

static_paths = st.subset_static(ij_bounds, dataset=var_ds, write_dir=static_write_dir)

pressure = read_pfb('/home/lp9617/workspace/conus2_spinups/runs/inputs/my_conus2_conus2_2006_WY_spinup/static/ss_pressure_head.pfb')
#saturation =
#WTD = hydro.calculate_water_table_depth(pressure, saturation, dz)
dz = (200, 100, 50, 25, 10, 5, 1, 0.6, 0.3, 0.1) #=392
total_z = sum(dz)
print("total_z, should be 392: ", total_z)
# WTD = total_z - 100 - pressure[0,...]
# WTD[WTD < 0.01] = 0.01
# print("WTD shape: ", WTD.shape)
# write_pfb('/home/lp9617/workspace/conus2_spinups/runs/inputs/my_conus2_conus2_2006_WY_spinup/static/ss_wtd.pfb', WTD)

# plt.figure()
# plt.imshow(WTD[:,:], origin='lower',norm=colors.LogNorm(vmin=0.01, vmax = 100))
# plt.colorbar()
# plt.savefig('/home/lp9617/workspace/conus2_spinups/runs/inputs/ss_wtd.png', dpi=1200)

print('Done!')
# os.chdir(static_write_dir)
# file_name = "pf_indicator.pfb"
# data = read_pfb(file_name)[7] 
# print(data.shape)

# plt.imshow(data, cmap="Reds", origin="lower")
# plt.colorbar()
# plt.title(file_name, fontsize = 14)
