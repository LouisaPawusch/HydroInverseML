import yaml
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parflow as pf
from parflow.tools.fs import mkdir, cp, get_absolute_path, exists
from parflow.tools.io import write_pfb, read_pfb
from parflow import Run
import parflow.tools.hydrology as hydro
import torch
import subsettools as st
from parflow.tools.settings import set_working_directory,get_working_directory
# from hf_hydrodata import register_api_pin

tcl_precision = 17

config_file = '/home/lp9617/workspace/pytorch/ML_Spinup_01xx/0122_3_settings.yaml'

file = open(config_file)
settings = yaml.load(file, Loader=yaml.FullLoader)

#static_inputs = '/projects/REEDMM/conus_ens/conus_ref'
#other_inputs = '/home/lp9617/workspace/data/further_converged/'
other_inputs = '/home/lp9617/workspace/data/' #das benutzen

#load all parameters from the settings file
testcase = 12
nr = 'c_73_12' # ******************************************************
#pressure_nr = "0111_pfrun_bp_ss_60_00"
print("working on ", nr)
saving_nr = settings['saving_nr']
seed = settings['seed']
n_epochs = settings['n_epochs']
# n_dims1 = settings['n_dims1']
# n_dims2 = settings['n_dims2']
# testcase = settings['testcase']
huc  = settings['huc']
huc_list = [huc]
grid = settings['grid']
# train_trunc = settings['train_trunc']

start = "2005-10-01"
wy = 2006
run_ds = "conus1_baseline_mod"
var_ds = "conus1_domain"

ML_saving_nr_string = saving_nr + "_sd" + str(seed) + "_" + str(n_epochs) + "_epochs"
test_indices = torch.load('/home/lp9617/workspace/pytorch/ML_Spinup_01xx/test_data/' + ML_saving_nr_string + '_test_indices.csv')
print('test_indices = ',test_indices)
test_indices_extreme = torch.load('/home/lp9617/workspace/pytorch/ML_Spinup_01xx/test_data/' + ML_saving_nr_string + '_test_indices_extreme.csv')
print('test_indices_extreme = ',test_indices_extreme)
testcase_idx = int(test_indices[testcase])
print('testcase_idx before shift = ',testcase_idx)
for extreme_idx in test_indices_extreme:
    if extreme_idx<=testcase_idx:
        testcase_idx += 1
print('testcase_idx_shifted = ',testcase_idx)


runname_helper = saving_nr + "_pfrun_" + nr
pressure_saving_nr_string = saving_nr + "_my_pressure_testcase_" + str(testcase)
my_runname = saving_nr + "_pfrun_" + nr

run_directory_path = '/home/lp9617/workspace/spinups/runs/outputs/'

# pressure_input_path = '/home/lp9617/workspace/spinups/runs/inputs/pressure_data/' #for m01 and c01 ******************************************************
pressure_input_path = '/home/lp9617/workspace/spinups/runs/inputs/pressure_data/' #for b01 ******************************************************
script_path           = '/home/lp9617/workspace/spinups/runs/run_scripts/'
run_path               = run_directory_path + my_runname
print("run path is:", run_path)

ij_bounds, mask = st.define_huc_domain(hucs=huc_list, grid=grid)
nj = ij_bounds[3] - ij_bounds[1]
ni = ij_bounds[2] - ij_bounds[0]
print("ij_bounds:", ij_bounds)
print("ni, nj:", ni, nj)


# flow_barrier_file      = 'UCRB_FlowBarrier.pfb'
script_file            = str(sys.argv[0]) #'generate_runscript_bp_ss_2.py' #*************************************************************************

#-----------------------------------------------------------------------------------------
# Setting up directories for run and copy inputs into it
#----------------------------------------------------------------------------------------- 

if os.path.exists(run_path) ==False: 
    os.makedirs(run_path)

print("working dir:",get_working_directory())
set_working_directory(run_path)
os.chdir(run_path)

mask_np = np.float64(mask)
write_pfb(run_path + '/mask.pfb', mask_np)
plt.figure()
plt.imshow(mask_np, origin='lower')
plt.savefig('mask.png')
print('mask has been saved')

init_pressure_file = 'const_pressurehead.pfb'
subsurface_file = 'pf_indicator_trunc.pfb'
slope_x_file = 'slope_x_trunc.pfb'
slope_y_file = 'slope_y_trunc.pfb'

cp(pressure_input_path + init_pressure_file)

indicator_whole = read_pfb('/home/lp9617/workspace/data/grid3d.v3.pfb')
indicator_trunc = indicator_whole[:,ij_bounds[1]:ij_bounds[3], ij_bounds[0]:ij_bounds[2]]
write_pfb(run_path + '/' + subsurface_file, indicator_trunc)
print('3')

slope_x_whole = read_pfb('/home/lp9617/workspace/data/slopex.pfb')
slope_x_trunc = slope_x_whole[:,ij_bounds[1]:ij_bounds[3], ij_bounds[0]:ij_bounds[2]]
write_pfb(run_path + '/' + slope_x_file, slope_x_trunc)
print('4')

slope_y_whole = read_pfb('/home/lp9617/workspace/data/slopey.pfb')
slope_y_trunc = slope_y_whole[:,ij_bounds[1]:ij_bounds[3], ij_bounds[0]:ij_bounds[2]]
write_pfb(run_path + '/' + slope_y_file, slope_y_trunc)
cp(script_path + script_file)

if testcase_idx < 50: #check, think
    print('we should not be here for testcase 12')
    print('check whether k-values are found with the following searching in the ensemble:')

    ksat_ens = pd.read_csv('/home/lp9617/workspace/spinups/ksat_ensemble_settings.csv')
    str_testcase = 'conus_K_ens.'+str(testcase_idx).zfill(3)
    filtered_row = ksat_ens[ksat_ens.iloc[:,0] == str_testcase]
    for i in range(1,10):
        print('filtered_row[',i,'] = ',filtered_row.iloc[0,i])

    test_features = torch.load('/home/lp9617/workspace/pytorch/ML_Spinup_01xx/test_data/0122_3_sd0_1500_epochs_test_features.csv').detach().numpy()
    print('test_features shape = ',test_features.shape)
    test_features_1 = test_features[testcase]
    test_ks_norm = test_features_1[0]
    minmax = torch.load('/home/lp9617/workspace/pytorch/ML_Spinup_01xx/test_data/0122_3_sd0_1500_epochs_minmaxs.csv')
    min_k = minmax[2,0].detach().numpy()
    max_k = minmax[2,1].detach().numpy()
    test_ks_unnorm = test_ks_norm * (max_k - min_k) + min_k

    print('test_ks shape = ',test_ks_unnorm.shape)
    print('np.unique(test_ks) = ',np.unique(test_ks_unnorm))


    init_pme_file = 'pme_trunc.pfb'
    pme_whole = read_pfb('/home/lp9617/workspace/data/PmE.flux.pfb')
    pme_trunc = pme_whole[:,ij_bounds[1]:ij_bounds[3], ij_bounds[0]:ij_bounds[2]]
    write_pfb(run_path + '/' + init_pme_file, pme_trunc)

    ksat_ens = pd.read_csv('/home/lp9617/workspace/spinups/ksat_ensemble_settings.csv')
    str_testcase = 'conus_K_ens.'+str(testcase_idx).zfill(3)
    filtered_row = ksat_ens[ksat_ens.iloc[:,0] == str_testcase]
    #pressure_saving_nr_string = 'pressure_k_trunc_150_'+ens_name
    #print("filtered row for testcase_idx_shifted= ",testcase_idx," = ", filtered_row)
    b1_perm = filtered_row['b1'].item()
    b2_perm = filtered_row['b2'].item()
    g1_perm = filtered_row['g1'].item()
    g2_perm = filtered_row['g2'].item()
    g3_perm = filtered_row['g3'].item()
    g4_perm = filtered_row['g4'].item()
    g5_perm = filtered_row['g5'].item()
    g6_perm = filtered_row['g6'].item()
    g7_perm = filtered_row['g7'].item()

else:
    #pressure_saving_nr_string = 'pressure_pet_trunc_150_'+str(i-50).zfill(3)

    z_dims = [100, 1.0, 0.6, 0.3, 0.1]
    n_dims3 = len(z_dims)
    print("n_dims3, should be 5 = ", n_dims3)

    init_pme_file = 'pme_trunc_150.pfb'
    #pme_input_path = '/home/lp9617/workspace/spinups/runs/inputs/pme_data/'

    PmE_ens = np.load('/home/lp9617/workspace/conus1_for_LP/pet_ens.npy')
    print("PmE_ens shape, should be (40,...) = ", PmE_ens.shape) #check, think

    huc_list = [huc]
    ij_bounds, mask = st.define_huc_domain(hucs=huc_list, grid='conus1')
    print("ij_bounds returns [imin, jmin, imax, jmax]")
    print(f"bounding box: {ij_bounds}")
    nj = ij_bounds[3] - ij_bounds[1]
    ni = ij_bounds[2] - ij_bounds[0]

    pme_trunc = PmE_ens[testcase_idx-50,ij_bounds[1]:ij_bounds[3], ij_bounds[0]:ij_bounds[2]] 
    print("pme_trunc shape = ", pme_trunc.shape)
    new_pme = np.zeros((n_dims3, nj, ni))
    for i in range(ni):
        for j in range(nj):
            new_pme[-1,j,i] = pme_trunc[j,i]*10
    
    print("new_pme shape = ", new_pme.shape)
    print("new_pme[4,50,50]  = ", new_pme[4,50,50]) #is ~10* than how it was from the training -npy
    write_pfb(run_path + '/'+init_pme_file, new_pme)

    test_features = torch.load('/home/lp9617/workspace/pytorch/ML_Spinup_01xx/test_data/0122_3_sd0_1500_epochs_test_features.csv').detach().numpy()
    print('test_features shape = ',test_features.shape)
    test_features_1 = test_features[testcase]
    test_pme_norm = test_features_1[1]
    minmax = torch.load('/home/lp9617/workspace/pytorch/ML_Spinup_01xx/test_data/0122_3_sd0_1500_epochs_minmaxs.csv')
    min_pme = minmax[1,0].detach().numpy()
    max_pme = minmax[1,1].detach().numpy()
    test_pme_unnorm = test_pme_norm * (max_pme - min_pme) + min_pme

    plt.figure()
    plt.imshow(new_pme[4,:,:]-test_pme_unnorm*10, origin='lower')
    plt.colorbar()
    plt.savefig('zz_pme_diff.png')

    b1_perm = 0.005
    b2_perm = 0.01
    g1_perm = 0.02
    g2_perm = 0.03
    g3_perm = 0.04
    g4_perm = 0.05
    g5_perm = 0.06
    g6_perm = 0.08
    g7_perm = 0.1


spinup_run = Run(my_runname, run_path)
#conus_ss = Run("conus_ss.sup")

spinup_run.FileVersion = 4

spinup_run.Process.Topology.P = 1
spinup_run.Process.Topology.Q = 1
spinup_run.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
spinup_run.ComputationalGrid.Lower.X = 0.0
spinup_run.ComputationalGrid.Lower.Y = 0.0
spinup_run.ComputationalGrid.Lower.Z = 0.0

spinup_run.ComputationalGrid.NX = ni
spinup_run.ComputationalGrid.NY = nj #check here, maybe other way round?
spinup_run.ComputationalGrid.NZ = 5

spinup_run.ComputationalGrid.DX = 1000.0
spinup_run.ComputationalGrid.DY = 1000.0
#"native" grid resolution is 2m everywhere X NZ=25 for 50m 
#computational domain.
spinup_run.ComputationalGrid.DZ = 100.0

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
spinup_run.GeomInput.Names = 'domaininput soilinput indi_input'

spinup_run.GeomInput.domaininput.GeomName = 'domain'
spinup_run.GeomInput.domaininput.InputType = 'Box'

spinup_run.GeomInput.soilinput.GeomName = 'soil'
spinup_run.GeomInput.soilinput.InputType = 'Box'

#---------------------------------------------------------
# Domain Geometry 
#---------------------------------------------------------
spinup_run.Geom.domain.Lower.X = 0.0
spinup_run.Geom.domain.Lower.Y = 0.0
spinup_run.Geom.domain.Lower.Z = 0.0
#  
spinup_run.Geom.domain.Upper.X = ni*1000 #3342000.0 #changed
spinup_run.Geom.domain.Upper.Y = nj*1000 #1888000.0
# this upper is synched to computational grid, not linked w/ Z multipliers
spinup_run.Geom.domain.Upper.Z = 500.0
spinup_run.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#---------------------------------------------------------
# Soil Geometry 
#---------------------------------------------------------
spinup_run.Geom.soil.Lower.X = 0.0
spinup_run.Geom.soil.Lower.Y = 0.0
spinup_run.Geom.soil.Lower.Z = 100.0
#  
spinup_run.Geom.soil.Upper.X = ni*1000 # 3342000.0 # changed
spinup_run.Geom.soil.Upper.Y = nj*1000# 1888000.0
# this upper is synched to computational grid, not linked w/ Z multipliers
spinup_run.Geom.soil.Upper.Z = 500.0

#-----------------------------------------------------------------------------
# Subsurface Indicator Geometry Input
#-----------------------------------------------------------------------------
spinup_run.GeomInput.indi_input.InputType = 'IndicatorField'
spinup_run.GeomInput.indi_input.GeomNames = 's1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8 b1 b2'
spinup_run.Geom.indi_input.FileName = subsurface_file # 'grid3d.v3.pfb' #changed, check that it is correct
spinup_run.dist(subsurface_file)

spinup_run.GeomInput.s1.Value = 1
spinup_run.GeomInput.s2.Value = 2
spinup_run.GeomInput.s3.Value = 3
spinup_run.GeomInput.s4.Value = 4
spinup_run.GeomInput.s5.Value = 5
spinup_run.GeomInput.s6.Value = 6
spinup_run.GeomInput.s7.Value = 7
spinup_run.GeomInput.s8.Value = 8
spinup_run.GeomInput.s9.Value = 9
spinup_run.GeomInput.s10.Value = 10
spinup_run.GeomInput.s11.Value = 11
spinup_run.GeomInput.s12.Value = 12
spinup_run.GeomInput.s13.Value = 13

spinup_run.GeomInput.g1.Value = 21
spinup_run.GeomInput.g2.Value = 22
spinup_run.GeomInput.g3.Value = 23
spinup_run.GeomInput.g4.Value = 24
spinup_run.GeomInput.g5.Value = 25
spinup_run.GeomInput.g6.Value = 26
spinup_run.GeomInput.g7.Value = 27
spinup_run.GeomInput.g8.Value = 28
spinup_run.GeomInput.b1.Value = 19
spinup_run.GeomInput.b2.Value = 20


#--------------------------------------------
# variable dz assignments
#------------------------------------------
spinup_run.Solver.Nonlinear.VariableDz = True
spinup_run.dzScale.GeomNames = 'domain'
spinup_run.dzScale.Type = 'nzList'
spinup_run.dzScale.nzListNumber = 5

# 5 layers, starts at 0 for the bottom to 5 at the top
# note this is opposite Noah/WRF
# layers are 0.1 m, 0.3 m, 0.6 m, 1.0 m, 100 m
spinup_run.Cell._0.dzScale.Value = 1.0
# 100 m * .01 = 1m 
spinup_run.Cell._1.dzScale.Value = 0.01
# 100 m * .006 = 0.6 m 
spinup_run.Cell._2.dzScale.Value = .006
# 100 m * 0.003 = 0.3 m 
spinup_run.Cell._3.dzScale.Value = .003
# 100 m * 0.001 = 0.1m = 10 cm which is default top Noah layer
spinup_run.Cell._4.dzScale.Value = 0.001

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

spinup_run.Geom.Perm.Names = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8 b1 b2'

# Values in m/hour

spinup_run.Geom.domain.Perm.Type = 'Constant'
spinup_run.Geom.domain.Perm.Value = 0.02

spinup_run.Geom.s1.Perm.Type = 'Constant'
spinup_run.Geom.s1.Perm.Value = 0.269022595

spinup_run.Geom.s2.Perm.Type = 'Constant'
spinup_run.Geom.s2.Perm.Value = 0.043630356

spinup_run.Geom.s3.Perm.Type = 'Constant'
spinup_run.Geom.s3.Perm.Value = 0.015841225

spinup_run.Geom.s4.Perm.Type = 'Constant'
spinup_run.Geom.s4.Perm.Value = 0.007582087

spinup_run.Geom.s5.Perm.Type = 'Constant'
spinup_run.Geom.s5.Perm.Value = 0.01818816

spinup_run.Geom.s6.Perm.Type = 'Constant'
spinup_run.Geom.s6.Perm.Value = 0.005009435

spinup_run.Geom.s7.Perm.Type = 'Constant'
spinup_run.Geom.s7.Perm.Value = 0.005492736

spinup_run.Geom.s8.Perm.Type = 'Constant'
spinup_run.Geom.s8.Perm.Value = 0.004675077

spinup_run.Geom.s9.Perm.Type = 'Constant'
spinup_run.Geom.s9.Perm.Value = 0.003386794

spinup_run.Geom.s10.Perm.Type = 'Constant'
spinup_run.Geom.s10.Perm.Value = 0.004783973

spinup_run.Geom.s11.Perm.Type = 'Constant'
spinup_run.Geom.s11.Perm.Value = 0.003979136

spinup_run.Geom.s12.Perm.Type = 'Constant'
spinup_run.Geom.s12.Perm.Value = 0.006162952

spinup_run.Geom.s13.Perm.Type = 'Constant'
spinup_run.Geom.s13.Perm.Value = 0.005009435

#perm(19) = 0.005
#perm(20) = 0.01
#perm(21) = 0.02
#perm(22) =  0.03
#perm(23) =  0.04
#perm(24) = 0.05
#perm(25) =  0.06
#perm(26) =  0.08
#perm(27) = 0.1
#perm(28) = 0.2
#


spinup_run.Geom.b1.Perm.Type = 'Constant'
spinup_run.Geom.b1.Perm.Value = b1_perm

spinup_run.Geom.b2.Perm.Type = 'Constant'
spinup_run.Geom.b2.Perm.Value = b2_perm

spinup_run.Geom.g1.Perm.Type = 'Constant'
spinup_run.Geom.g1.Perm.Value = g1_perm

spinup_run.Geom.g2.Perm.Type = 'Constant'
spinup_run.Geom.g2.Perm.Value = g2_perm

spinup_run.Geom.g3.Perm.Type = 'Constant'
spinup_run.Geom.g3.Perm.Value = g3_perm

spinup_run.Geom.g4.Perm.Type = 'Constant'
spinup_run.Geom.g4.Perm.Value = g4_perm

spinup_run.Geom.g5.Perm.Type = 'Constant'
spinup_run.Geom.g5.Perm.Value = g5_perm

spinup_run.Geom.g6.Perm.Type = 'Constant'
spinup_run.Geom.g6.Perm.Value = g6_perm

spinup_run.Geom.g7.Perm.Type = 'Constant'
spinup_run.Geom.g7.Perm.Value = g7_perm

spinup_run.Geom.g8.Perm.Type = 'Constant'
spinup_run.Geom.g8.Perm.Value = 0.2


spinup_run.Perm.TensorType = 'TensorByGeom'

spinup_run.Geom.Perm.TensorByGeom.Names = 'domain'

spinup_run.Geom.domain.Perm.TensorValX = 1.0
spinup_run.Geom.domain.Perm.TensorValY = 1.0
spinup_run.Geom.domain.Perm.TensorValZ = 1.0


#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

spinup_run.SpecificStorage.Type = 'Constant'
spinup_run.SpecificStorage.GeomNames = 'domain'
spinup_run.Geom.domain.SpecificStorage.Value = 1.0e-5
#pfset Geom.domain.SpecificStorage.Value 0.0

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

spinup_run.Phase.Names = 'water'

spinup_run.Phase.water.Density.Type = 'Constant'
spinup_run.Phase.water.Density.Value = 1.0

spinup_run.Phase.water.Viscosity.Type = 'Constant'
spinup_run.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

spinup_run.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

spinup_run.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

spinup_run.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

# 
spinup_run.TimingInfo.BaseUnit = 10.0
spinup_run.TimingInfo.StartCount = 0
spinup_run.TimingInfo.StartCount = 0 #0 #11 #changed
spinup_run.TimingInfo.StartTime = 0.0
spinup_run.TimingInfo.DumpInterval = 5000 #5000 #changed (deleted a 0)
spinup_run.TimingInfo.StopTime = 1000000 #5000000 # 5000000.0 #changed (added a 0)

spinup_run.TimeStep.Type = 'Constant'
spinup_run.TimeStep.Value = 0.1
spinup_run.TimeStep.Value = 100.0
#pfset TimeStep.Value             0.5

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

spinup_run.Geom.Porosity.GeomNames = 'domain'

spinup_run.Geom.domain.Porosity.Type = 'Constant'
spinup_run.Geom.domain.Porosity.Value = 0.25

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

spinup_run.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

spinup_run.Phase.RelPerm.Type = 'VanGenuchten'
spinup_run.Phase.RelPerm.GeomNames = 'domain'

spinup_run.Geom.domain.RelPerm.Alpha = 0.3
spinup_run.Geom.domain.RelPerm.N = 3.

spinup_run.Geom.soil.RelPerm.Alpha = 1.
spinup_run.Geom.soil.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

spinup_run.Phase.Saturation.Type = 'VanGenuchten'
spinup_run.Phase.Saturation.GeomNames = 'domain'

spinup_run.Geom.domain.Saturation.Alpha = 0.3
spinup_run.Geom.domain.Saturation.N = 3.
spinup_run.Geom.domain.Saturation.SRes = 0.1
spinup_run.Geom.domain.Saturation.SSat = 1.0

spinup_run.Geom.soil.Saturation.Alpha = 1.0
spinup_run.Geom.soil.Saturation.N = 2.
spinup_run.Geom.soil.Saturation.SRes = 0.1
spinup_run.Geom.soil.Saturation.SSat = 1.0


#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
spinup_run.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

spinup_run.Cycle.Names = 'constant'
spinup_run.Cycle.constant.Names = 'alltime'
spinup_run.Cycle.constant.alltime.Length = 10000000
spinup_run.Cycle.constant.Repeat = -1

#  
#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
spinup_run.BCPressure.PatchNames = 'x_lower x_upper y_lower y_upper z_lower z_upper'

spinup_run.Patch.x_lower.BCPressure.Type = 'FluxConst'
spinup_run.Patch.x_lower.BCPressure.Cycle = 'constant'
spinup_run.Patch.x_lower.BCPressure.alltime.Value = 0.0

spinup_run.Patch.y_lower.BCPressure.Type = 'FluxConst'
spinup_run.Patch.y_lower.BCPressure.Cycle = 'constant'
spinup_run.Patch.y_lower.BCPressure.alltime.Value = 0.0

spinup_run.Patch.z_lower.BCPressure.Type = 'FluxConst'
spinup_run.Patch.z_lower.BCPressure.Cycle = 'constant'
spinup_run.Patch.z_lower.BCPressure.alltime.Value = 0.0

spinup_run.Patch.x_upper.BCPressure.Type = 'FluxConst'
spinup_run.Patch.x_upper.BCPressure.Cycle = 'constant'
spinup_run.Patch.x_upper.BCPressure.alltime.Value = 0.0

spinup_run.Patch.y_upper.BCPressure.Type = 'FluxConst'
spinup_run.Patch.y_upper.BCPressure.Cycle = 'constant'
spinup_run.Patch.y_upper.BCPressure.alltime.Value = 0.0

## overland flow boundary condition with very heavy rainfall then slight ET
#conus_ss.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
spinup_run.Patch.z_upper.BCPressure.Type = 'SeepageFace'
#conus_ss.Patch.z_upper.BCPressure.Type = 'FluxConst'
spinup_run.Patch.z_upper.BCPressure.Cycle = 'constant'
spinup_run.Patch.z_upper.BCPressure.alltime.Value = 0.0

# PmE flux 
spinup_run.Solver.EvapTransFile = True
spinup_run.Solver.EvapTrans.FileName =  init_pme_file 
spinup_run.dist(init_pme_file)


#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

spinup_run.TopoSlopesX.Type = 'PFBFile'
spinup_run.TopoSlopesX.GeomNames = 'domain'
spinup_run.TopoSlopesX.FileName = slope_x_file 
spinup_run.ComputationalGrid.NZ = 1
spinup_run.dist(slope_x_file)
spinup_run.ComputationalGrid.NZ = 5
#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

spinup_run.TopoSlopesY.Type = 'PFBFile'
spinup_run.TopoSlopesY.GeomNames = 'domain'
spinup_run.TopoSlopesY.FileName =  slope_y_file 
spinup_run.ComputationalGrid.NZ = 1
spinup_run.dist(slope_y_file)
spinup_run.ComputationalGrid.NZ = 5

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

spinup_run.Mannings.Type = 'Constant'
spinup_run.Mannings.GeomNames = 'domain'
spinup_run.Mannings.Geom.domain.Value = 1.0e-5


#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

spinup_run.PhaseSources.water.Type = 'Constant'
spinup_run.PhaseSources.water.GeomNames = 'domain'
spinup_run.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

spinup_run.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

spinup_run.Solver = 'Richards'
spinup_run.Solver.MaxIter = 50000

spinup_run.Solver.TerrainFollowingGrid = True


spinup_run.Solver.Nonlinear.MaxIter = 150
spinup_run.Solver.Nonlinear.ResidualTol = 1e-5
spinup_run.Solver.Nonlinear.EtaValue = 0.0001


spinup_run.Solver.PrintSubsurf = False
spinup_run.Solver.AbsTol = 1E-10


#pfset Solver.Nonlinear.EtaChoice                         EtaConstant
#Solver.Nonlinear.EtaValue                          0.01
spinup_run.Solver.Nonlinear.UseJacobian = True
#pfset Solver.Nonlinear.UseJacobian                       False 
spinup_run.Solver.Nonlinear.DerivativeEpsilon = 1e-16
spinup_run.Solver.Nonlinear.StepTol = 1e-20
spinup_run.Solver.Nonlinear.Globalization = 'LineSearch'
spinup_run.Solver.Linear.KrylovDimension = 250
spinup_run.Solver.Linear.MaxRestarts = 2

spinup_run.Solver.Linear.Preconditioner = 'MGSemi'
#conus_ss.Solver.Linear.Preconditioner = 'PFMG'
#conus_ss.Solver.Linear.Preconditioner = 'SMG'
#conus_ss.Solver.Linear.Preconditioner.PCMatrixType = 'FullJacobian'
#pfset OverlandFlowDiffusive  1


spinup_run.Solver.WriteSiloSubsurfData = False
spinup_run.Solver.WriteSiloPressure = False
spinup_run.Solver.WriteSiloSaturation = False
spinup_run.Solver.WriteSiloConcentration = False
spinup_run.Solver.WriteSiloSlopes = False
spinup_run.Solver.WriteSiloMask = False

spinup_run.Solver.PrintSubsurfData = True 
spinup_run.Solver.PrintSpecificStorage = False
spinup_run.Solver.PrintMask = False
spinup_run.Solver.PrintPressure = True 
spinup_run.Solver.PrintSaturation = True 

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
spinup_run.ICPressure.Type = 'HydroStaticPatch'
spinup_run.ICPressure.GeomNames = 'domain'
spinup_run.Geom.domain.ICPressure.Value = 0.0
# really WT is -4 BLS
#pfset Geom.domain.ICPressure.Value                      -4.0


spinup_run.Geom.domain.ICPressure.RefGeom = 'domain'
spinup_run.Geom.domain.ICPressure.RefPatch = 'z_lower'

# restart from last timestep 
spinup_run.ICPressure.Type = 'PFBFile'
spinup_run.ICPressure.GeomNames = 'domain'
spinup_run.Geom.domain.ICPressure.FileName =  init_pressure_file # 'press.in.pfb'
spinup_run.dist(init_pressure_file)

#spinup key
# True=skim pressures, False = regular (default)
#pfset Solver.Spinup           True
#conus_ss.Solver.Spinup = True

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------


spinup_run.run()
# """