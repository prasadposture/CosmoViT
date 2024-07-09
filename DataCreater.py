import numpy as np
from Convolver import file_reader, fld_rdr
import os

"""
Requires
Redshift array
Path of redshifts file
Path where maps are stored
path where you want to store your outputs

Gives
halo, density, HI maps in terms of numpy array text file for later uses
"""
if not os.path.exists('./Data'):
    os.mkdir('./Data')


# Reading from the redshifts
redshifts = np.flip(np.linspace(0.1, 12, 30))  # till 3 decimals
np.save("emulator/Data/redshifts.npy", redshifts)

# Initialize lists to store the arrays
Halo_maps = []
DM_maps = []
HI_maps = []
HI_maprs = []

for zi in redshifts:
    Halo_arr = file_reader(f'128_007_PINION_model/128_007_PINION_model/Halo_map_{zi:.3f}')
    DM_arr = file_reader(f'128_007_PINION_model/128_007_PINION_model/DM_map_{zi:.3f}')
    HI_arr = fld_rdr(f"128_007_PINION_model/128_007_PINION_model/ionz_out/HI_map_{zi:.3f}")
    HIrs_arr = fld_rdr(f"128_007_PINION_model/128_007_PINION_model/ionz_out/HI_maprs_{zi:.3f}")

    Halo_maps.append(Halo_arr)
    DM_maps.append(DM_arr)
    HI_maps.append(HI_arr)
    HI_maprs.append(HIrs_arr)

# Convert lists to numpy arrays
Halo_maps = np.array(Halo_maps)
DM_maps = np.array(DM_maps)
HI_maps = np.array(HI_maps)
HI_maprs = np.array(HI_maprs)

print(Halo_maps.shape)
np.save("emulator/Data/HaloData.npy", Halo_maps)
print("HaloData saved")
np.save("emulator/Data/DMData.npy", DM_maps)
print("DMData saved")
print(HI_maprs.shape)
np.save("emulator/Data/HIData.npy", HI_maps)
print("HIData saved")
np.save("emulator/Data/HIRSData.npy", HI_maprs)
print("HIRSData saved")
