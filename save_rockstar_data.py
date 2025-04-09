import numpy as np
import matplotlib.pyplot as plt
import h5py
from pyHalo.PresetModels.wdm import WDM
import os
import re
import time
import sys

h = 0.7
dm_model = sys.argv[1]

# READING IN ROCKSTAR DATA
#rootdir = '/carnegie/nobackup/groups/dmtheory/mergerTrees/Symphony/MilkyWay/resolutionX1/CDM'
rootdir = '/carnegie/nobackup/groups/dmtheory/mergerTrees/COZMIC/MilkyWay/resolutionX1/WDM:6keV'

# Creating the hdf5 file that we're reading data into
with h5py.File('symphony_data_' + dm_model + '.hdf5', 'w') as f:
    num_cols = 8
    maxshape = (None,)
    f.create_dataset("id", maxshape=maxshape, shape=(0,), dtype='f8', chunks=True)
    f.create_dataset("pid", maxshape=maxshape, shape=(0,), dtype='f8', chunks=True)
    f.create_dataset("Mvir", maxshape=maxshape, shape=(0,), dtype='f8', chunks=True)
    f.create_dataset("Rvir", maxshape=maxshape, shape=(0,), dtype='f8', chunks=True)
    f.create_dataset("rs", maxshape=maxshape, shape=(0,), dtype='f8', chunks=True)
    f.create_dataset("x", maxshape=maxshape, shape=(0,), dtype='f8', chunks=True)
    f.create_dataset("y", maxshape=maxshape, shape=(0,), dtype='f8', chunks=True)
    f.create_dataset("z", maxshape=maxshape, shape=(0,), dtype='f8', chunks=True)
    f.create_dataset("halo_file", maxshape=maxshape, shape=(0,), dtype='i8', chunks=True)

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith('tree_0_0_0.dat'):
            subdir_digits =  re.sub(r"\D", "", subdir)
            rock_realization = np.loadtxt(os.path.join(subdir, file), skiprows = 50, usecols = (0, 1, 5, 10, 11, 12, 17, 18, 19))

            for i in range(len(rock_realization[0])):
                if i > 2:
                    rock_realization[:,i] = rock_realization[:,i]/h

            current = (rock_realization[:,0] == 1)
            halo_array = np.array([int(subdir_digits[1:]) for _ in range(len(rock_realization[:,0][current]))])

            new_data = {
                "id": rock_realization[:,1][current],
                "pid": rock_realization[:, 2][current],
                "Mvir": rock_realization[:, 3][current],
                "Rvir": rock_realization[:, 4][current]/1000, # So that all distance scales are in units of Mpc
                "rs": rock_realization[:, 5][current],
                "x": rock_realization[:, 6][current],
                "y": rock_realization[:, 7][current],
                "z": rock_realization[:, 8][current],
                "halo_file": halo_array
                }

            with h5py.File('symphony_data_' + dm_model + '.hdf5', 'a') as f:
                 for key, data in new_data.items():
                     dataset = f[key]
                     old_size = dataset.shape[0]
                     new_size = old_size + len(data)

                     # Resize and append data
                     dataset.resize((new_size,))
                     dataset[old_size:new_size] = data

print('code executed!')
