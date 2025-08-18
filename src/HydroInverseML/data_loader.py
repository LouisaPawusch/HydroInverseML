import torch
import hf_hydrodata as hf
from HydroInverseML.src.HydroInverseML.hf_api_pin import return_api_pin

email, pin = return_api_pin()
hf.register_api_pin(email, pin)
# hf.register_api_pin("<your_email>", "<your_pin>")


def open_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def access_hydrodata():
    # ss_water_table_depth only exists in conus2_domain (=CONUS2.0)
    
    #CONUS2.1 data
    # pf_output_data = hf.get_gridded_data({"dataset": "conus2_baseline", "variable":"water_table_depth", "grid": "conus2"})

    #CONUS2.0 data
    # Note: ss_water_table_depth is at this time only available in conus2_domain
    options = {
      "dataset": "conus2_domain", "grid": "conus2"
    }
    my_variables = ["pme", "pf_indicator", "slope_x", "slope_y", "elevation", "ss_water_table_depth"]

    hf.get_gridded_files(options, variables=my_variables)

    return True