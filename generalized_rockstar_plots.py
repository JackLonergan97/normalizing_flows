import numpy as np
import matplotlib.pyplot as plt
import h5py
from pyHalo.PresetModels.wdm import WDM
import xml.etree.ElementTree as ET
import sys

dm_model = sys.argv[1]

# Reading in z = 0 symphony data
f = h5py.File('symphony_data_' + dm_model + '.hdf5', 'r')
ids = f['id'][:]
pid = f['pid'][:]
Mvir = f['Mvir'][:]
Rvir = f['Rvir'][:]
rs = f['rs'][:]
x = f['x'][:]
y = f['y'][:]
z = f['z'][:]
halo_file = f['halo_file'][:]
unique_file = np.unique(halo_file)

# Extracting Milky Way host halo ID's from a separate .xml file
tree = ET.parse('/home/jlonergan/Galacticus/galacticus/constraints/pipelines/darkMatter/symphonyZoomInHostHaloIDs.xml')
root = tree.getroot()
MW_dict = root[1].attrib
HaloIDs = np.array([])
for value in MW_dict.values():
    HaloIDs = np.append(HaloIDs, float(value))

masses = np.array([])

for i in range(len(unique_file)):
#for i in range(6):
    MW = (Mvir > 1e12) & (Mvir < 2e12) & (halo_file == unique_file[i])
    halos = (halo_file == unique_file[i])
    for j in range(len(ids[MW])):
        if dm_model == 'CDM':
            if ids[MW][j] in HaloIDs:
                MW_Mvir = Mvir[MW][j]
                MW_Rvir = Rvir[MW][j]
                MW_x = x[MW][j]
                MW_y = y[MW][j]
                MW_z = z[MW][j]

                radii = np.sqrt((x[halos] - MW_x)**2 + (y[halos] - MW_y)**2 + (z[halos] - MW_z)**2)
                subhalos = (pid[halos] > 1) & (radii < MW_Rvir)

                masses = np.append(masses, Mvir[halos][subhalos])
        else:
             MW_Mvir = Mvir[MW][j]
             MW_Rvir = Rvir[MW][j]
             MW_x = x[MW][j]
             MW_y = y[MW][j]
             MW_z = z[MW][j]
    
             radii = np.sqrt((x[halos] - MW_x)**2 + (y[halos] - MW_y)**2 + (z[halos] - MW_z)**2)
             subhalos = (pid[halos] > 1) & (radii < MW_Rvir)

             masses = np.append(masses, Mvir[halos][subhalos])
        

    #total_radii.append(np.sqrt((x[halo_clip] - MW_x[i])**2 + (y[halo_clip] - MW_y[i])**2 + (z[halo_clip] - MW_z[i])**2))

# Making Galacticus subhalo clips
galResolution = 1e6
countTree = 450

f_cdm = h5py.File('gal_outputs/darkMatterOnlySubHalosWDM_MW.hdf5', 'r')
cdm_massInfall = f_cdm['Outputs/Output1/nodeData/basicMass'][:]
cdm_concentration = f_cdm['Outputs/Output1/nodeData/concentration'][:]
cdm_isCentral = f_cdm['Outputs/Output1/nodeData/nodeIsIsolated'][:]
cdm_weight = f_cdm['Outputs/Output1/nodeData/nodeSubsamplingWeight'][:]
cdm_treeIndex = f_cdm['Outputs/Output1/nodeData/mergerTreeIndex'][:]
cdm_massBound = f_cdm['Outputs/Output1/nodeData/satelliteBoundMass'][:]
gal_cdm_x = f_cdm['Outputs/Output1/nodeData/positionOrbitalX'][:]
gal_cdm_y = f_cdm['Outputs/Output1/nodeData/positionOrbitalY'][:]
gal_cdm_z = f_cdm['Outputs/Output1/nodeData/positionOrbitalZ'][:]
gal_cdm_radii = np.sqrt(gal_cdm_x**2 + gal_cdm_y**2 + gal_cdm_z**2)
gal_cdm_vmax = f_cdm['Outputs/Output1/nodeData/darkMatterProfileDMOVelocityMaximum'][:]
cdm_radiusVirial = f_cdm['Outputs/Output1/nodeData/darkMatterOnlyRadiusVirial'][:]
cdm_centrals = (cdm_isCentral[:] == 1)
min_cdm_rvir = min(cdm_radiusVirial[cdm_centrals])
gal_cdm_subhalos = (cdm_isCentral == 0) & (cdm_massInfall > 2 * galResolution) & (gal_cdm_radii <= min_cdm_rvir) & (gal_cdm_vmax > 8) # the (cdm_massBound/cdm_Mhost > 5e-5) condition is so I can make the leftmost end of the ratio plot. In that plot, I make a vertical line showing the convergence limit of M_sub/M_host < 2.7e-4
gal_cdm_masses = cdm_massBound[gal_cdm_subhalos]

# Making Rockstar SMFs
m = np.geomspace(min(masses), max(masses), 100)
cdm_halos = np.cumsum(np.histogram(masses,m)[0][::-1])[::-1]/len(unique_file)
gal_cdm_halos = np.cumsum(np.histogram(gal_cdm_masses, m)[0][::-1])[::-1]/countTree

plt.figure()
plt.plot(m[:-1], cdm_halos, 'k-', label = 'WDM (Rockstar)')
#plt.plot(m[:-1], wdm_halos, 'r-', label = 'WDM (Rockstar)')
plt.plot(m[:-1], gal_cdm_halos, 'k--', label = 'WDM (Galacticus)')
#plt.plot(m[:-1], gal_wdm_halos, 'r--', label = 'WDM (Galacticus)')
plt.axvline(x = 2.7e-4, ymin = 1e-3, ymax = 1e5, linestyle = '--', color = 'grey')
#plt.xlim(5e-5, 2e-1)
plt.xlim(1e7, 1e12)
#plt.ylim(1, 3e2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$ M/M_{host} $')
plt.ylabel('$ N(> M) $')
plt.title('Bound SMFs')
plt.legend()
plt.savefig('plots/rockstar_smf.png')

# Making ratio plot
plt.figure()
plt.plot(m[:-1]/MW_Mvir, cdm_halos/gal_cdm_halos, 'k-')
plt.xscale('log')
plt.xlim(5e-5, 0.1)
#plt.ylim(0, 2)
plt.xlabel('$ M_{sub}/M_{host} $')
plt.ylabel('$ N_{sub}(> M_{sub}/M_{host}) $ ratio')
plt.title('Symphony/Galacticus')
plt.savefig('plots/symphony_galacticus_ratios.png')

print('code executed!')
