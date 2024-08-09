import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pickle
import sys
import itertools
import warnings
warnings.filterwarnings('ignore')

resolution = sys.argv[1]
n_files = 100
KPC_TO_MPC = 1e-3

# Create the summary statistic distribution plot

standard_CDM_output_path = os.getcwd() + '/standard_inference_output_CDM_res' + resolution + '/'
#standard_WDM_output_path = os.getcwd() + '/standard_inference_output_WDM_res' + resolution + '/'
emulator_CDM_output_path = os.getcwd() + '/emulator_inference_output_CDM_res' + resolution + '_annulus/'
#emulator_WDM_output_path = os.getcwd() + '/emulator_inference_output_WDM_res' + resolution + '/'
#dan_CDM_output_path = os.getcwd() + '/dan_inference_output/'

#standard_CDM_output_path = os.getcwd() + '/cdm_test/'
#emulator_CDM_output_path = os.getcwd() + '/em_test/'
#dan_CDM_output_path = os.getcwd() + '/em_test/'

standard_CDM_statistic = np.array([])
standard_WDM_statistic = np.array([])
emulator_CDM_statistic = np.array([])
emulator_WDM_statistic = np.array([])
#dan_CDM_statistic = np.array([])

for i in range(1, n_files+1):
#    stat = np.loadtxt(emulator_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,-2]
    try:
        stat = np.loadtxt(standard_CDM_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,-2]
        standard_CDM_statistic = np.append(standard_CDM_statistic, stat)

        if len(stat) < 100:
            print('CDM job_' + str(i) + ' only has ' + str(len(stat)) + ' simulation outputs') 

        #stat = np.loadtxt(standard_WDM_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,-2]
        #standard_WDM_statistic = np.append(standard_WDM_statistic, stat)

        stat = np.loadtxt(emulator_CDM_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,-2]
        emulator_CDM_statistic = np.append(emulator_CDM_statistic, stat)

        if len(stat) < 100:
            print('emulator job_' + str(i) + ' only has ' + str(len(stat)) + ' simulation outputs')

        #stat = np.loadtxt(emulator_WDM_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,-2]
        #emulator_WDM_statistic = np.append(emulator_WDM_statistic, stat)
    
        #stat = np.loadtxt(dan_CDM_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,-2]
        #dan_CDM_statistic = np.append(dan_CDM_statistic, stat)
    except:
        print('incorrect path name used!')
        pass

#plt.hist(dan_CDM_statistic, bins = 8, range = (0,2), label = 'Daniel\'s Model', fill = False, edgecolor = 'blue')
#plt.hist(dan_CDM10_statistic, bins = 8, range = (0,2), label = 'Emulator', fill = False, edgecolor = 'orange')
#plt.xlabel('$S_{lens}$')
#plt.legend()
#plt.savefig('plots/hist_stat.png')

# Making the downward curved plot
s = np.linspace(0, 10., 10000)
x_o_CDM = []
x_o_WDM = []
x_e_CDM = []
x_e_WDM = []
x_d_CDM = []

for xi in s:
    n = np.sum(standard_CDM_statistic > xi)
    x_o_CDM.append(n)

    #n = np.sum(standard_WDM_statistic > xi)
    #x_o_WDM.append(n)

    n = np.sum(emulator_CDM_statistic > xi)
    x_e_CDM.append(n)

    #n = np.sum(emulator_WDM_statistic > xi)
    #x_e_WDM.append(n)

    #n = np.sum(dan_CDM_statistic > xi)
    #x_d_CDM.append(n)

x_o_CDM = np.array(x_o_CDM)/len(standard_CDM_statistic)
#x_o_WDM = np.array(x_o_WDM)/len(standard_WDM_statistic)
x_e_CDM = np.array(x_e_CDM)/len(emulator_CDM_statistic)
#x_e_WDM = np.array(x_e_WDM)/len(emulator_WDM_statistic)
#x_d_CDM = np.array(x_d_CDM)/len(dan_CDM_statistic)

#plt.plot(s, x_o_CDM, '-', color = 'blue', label = 'CDM')
#plt.plot(s, x_o_WDM, 'o', color = 'blue', label = 'WDM')
#plt.plot(s, x_e_CDM, '-', color = 'orange', label = 'CDMEmulator')
#plt.plot(s, x_e_WDM, 'o', color = 'orange', label = 'WDMEmulator')
#plt.legend()
#plt.savefig('plots/dm_ssd.png')

# Making the plot which starts at (0,0) and ends at (1,1)
x_o_CDM = []
x_o_WDM = []
x_e_CDM = []
x_e_WDM = []
x_d_CDM = []

for xi in s:
    n = np.sum(standard_CDM_statistic < xi)
    x_o_CDM.append(n)

    #n = np.sum(standard_WDM_statistic < xi)
    #x_o_WDM.append(n)

    n = np.sum(emulator_CDM_statistic < xi)
    x_e_CDM.append(n)

    #n = np.sum(emulator_WDM_statistic < xi)
    #x_e_WDM.append(n)

    #n = np.sum(dan_CDM_statistic < xi)
    #x_d_CDM.append(n)

x_o_CDM = np.array(x_o_CDM)/len(standard_CDM_statistic)
#print('length of x_o_CDM: ' + str(len(x_o_CDM)))
#x_o_WDM = np.array(x_o_WDM)/len(standard_WDM_statistic)
#print('length of x_o_WDM: ' + str(len(x_o_WDM)))
x_e_CDM = np.array(x_e_CDM)/len(emulator_CDM_statistic)
#print('length of x_e_CDM: ' + str(len(x_e_CDM)))
#x_e_WDM = np.array(x_e_WDM)/len(emulator_WDM_statistic)
#print('length of x_e_WDM: ' + str(len(x_e_WDM)))
#x_d_CDM = np.array(x_d_CDM)/len(dan_CDM_statistic)

#plt.plot(s, x_o_WDM/x_o_CDM, label = 'Original')
#plt.plot(s, x_e_WDM/x_e_CDM, label = 'Emulator')
#plt.xlim((0, 0.1))
#plt.xlabel('$S_{lens}$')
#plt.ylabel('WDM to CDM Ratio')
#plt.legend()
#plt.savefig('plots/likelihood_ratio_res' + resolution + '.png')

fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.set_size_inches(17, 5)

ax1.plot(s, x_o_CDM, 'k-', label = 'CDM')
ax1.plot(s, x_e_CDM, 'r-', label = 'Emulator')
#ax1.plot(s, x_o_WDM, 'k-.', label = 'many subhalos')
#ax1.plot(s, x_e_WDM, 'r-.', label = 'many subhalos')
ax1.set_xlim([0, 2])
ax1.set_xlabel('$S_{lens}$')
ax1.set_ylabel('Fraction < $S_{lens}$')
ax1.legend()
ax2.plot(s, x_o_CDM, 'k-', label = 'CDM')
ax2.plot(s, x_e_CDM, 'r-', label = 'Emulator')
#ax2.plot(s, x_o_WDM, 'k-.', label = 'many subhalos')
#ax2.plot(s, x_e_WDM, 'r-.', label = 'many subhalos')
ax2.set_xlim([0, 0.1])
ax2.set_ylim([0, 0.1])
ax2.set_xlabel('$S_{lens}$')
ax2.set_ylabel('Fraction < $S_{lens}$')
ax2.legend()
plt.savefig('plots/inverted_dm_ssd_res' + resolution + '.png')

f_dan = h5py.File('cdm_subhalo_data.hdf5', 'r')
#dan_m_infall = f_dan['m_infall'][:]
dan_c = f_dan['c'][:]
#dan_m_bound = f_dan['m_bound'][:]
#dan_z_infall = f_dan['z_infall'][:]
#dan_r3d = f_dan['r3d'][:]
#dan_rt = f_dan['rt'][:]
#dan_x = f_dan['x'][:]
#dan_y = f_dan['y'][:]
#dan_alpha = f_dan['alpha_Rs'][:]
f_dan.close()

f_em = h5py.File('bad_em_subhalo_data.hdf5', 'r')
#em_m_infall = f_em['m_infall'][:]
em_c = f_em['c'][:]
#em_m_bound = f_em['m_bound'][:]
#em_z_infall = f_em['z_infall'][:]
#em_r3d = f_em['r3d'][:]
#em_rt = f_em['rt'][:]
#em_x = f_em['x'][:]
#em_y = f_em['y'][:]
#em_alpha = f_em['alpha_Rs'][:]
f_em.close()

f, axes = plt.subplots(1, 2)
f.set_size_inches(20,12)
axes[0].hist(dan_c, bins = 20, range = (0, np.max(em_c)), label = 'CDM', fill = True, edgecolor = 'blue')
axes[0].set(title = 'Concentrations over Emulator range')
axes[0].hist(em_c, bins = 20, range = (0, np.max(em_c)), label = 'Emulator', fill = False, edgecolor = 'orange')
axes[0].set(title = 'Concentrations over Emulator range')
axes[1].hist(dan_c, bins = 20, range = (0, np.max(dan_c)), label = 'CDM', fill = True, edgecolor = 'blue')
axes[1].set(title = 'Concentrations over Galacticus range')
axes[1].hist(em_c, bins = 20, range = (0, np.max(dan_c)), label = 'Emulator', fill = False, edgecolor = 'orange')
axes[1].set(title = 'Concentrations over Galacticus range')
plt.savefig('plots/concentrations.png')

MPC_TO_KPC = 1e3
rhoc = 277536627245.83075 # got from lens_cosmo.rhoc in another script [Mpc]
kpc_per_arcsec = 5.013831007195271 # at z_lens = 0.34

f, axes = plt.subplots(3,3)
f.set_size_inches(15, 20)

labels = ['CDM', 'Emulator']
#labels = ['no annulus', 'annulus']
#labels = ['low S_lens', 'high S_lens']

#labels = ['CDM', 'emulator (low S_lens)', 'emulator (high S_lens)']
#labels = ['no annulus', 'annulus (low S_lens)', 'annulus (high S_lens)']

axes[0, 0].hist(dan_m_infall, bins = 20, range = (0, 1e10), label = labels[0], fill = True, edgecolor = 'blue')
axes[0, 0].hist(em_m_infall, bins = 20, range = (0, 1e10), label = labels[1], fill = False, edgecolor = 'orange')
axes[0, 0].set(yscale = 'log', title = 'infall mass')
axes[0, 0].legend()
axes[0, 1].hist(dan_c, bins = 20, label = labels[0], fill = True, edgecolor = 'blue')
axes[0, 1].hist(em_c, bins = 20, label = labels[1], fill = False, edgecolor = 'orange')
axes[0, 1].set(yscale = 'log', title = 'concentration')
axes[0, 1].legend()
axes[0, 2].hist(dan_m_bound, bins = 20, range = (0,1e10), label = labels[0], fill = True, edgecolor = 'blue')
axes[0, 2].hist(em_m_bound, bins = 20, range = (0,1e10), label = labels[1], fill = False, edgecolor = 'orange')
axes[0, 2].set(yscale = 'log', title = 'bound mass')
axes[0, 2].legend()
axes[1, 0].hist(dan_z_infall, bins = 20, label = labels[0], fill = True, edgecolor = 'blue')
axes[1, 0].hist(em_z_infall, bins = 20, label = labels[1], fill = False, edgecolor = 'orange')
axes[1, 0].set(yscale = 'log', title = 'infall redshift')
axes[1, 0].legend()
axes[1, 1].hist(dan_r3d, bins = 20, label = labels[0], fill = True, edgecolor = 'blue')
axes[1, 1].hist(em_r3d, bins = 20, label = labels[1], fill = False, edgecolor = 'orange')
axes[1, 1].set(yscale = 'log', title = 'orbital radius')
axes[1, 1].legend()
axes[1, 2].hist(dan_rt, bins = 20, range = (0, np.max(dan_rt)), label = labels[0], fill = True, edgecolor = 'blue')
axes[1, 2].hist(em_rt, bins = 20, range = (0, np.max(dan_rt)), label = labels[1], fill = False, edgecolor = 'orange')
axes[1, 2].set(yscale = 'log', title = 'truncation radius')
axes[1, 2].legend()
axes[2, 0].hist(dan_x, bins = 20, label = labels[0], fill = True, edgecolor = 'blue')
axes[2, 0].hist(em_x, bins = 20, label = labels[1], fill = False, edgecolor = 'orange')
axes[2, 0].set(yscale = 'log', title = 'x position')
axes[2, 0].legend()
axes[2, 1].hist(dan_y, bins = 20, label = labels[0], fill = True, edgecolor = 'blue')
axes[2, 1].hist(em_y, bins = 20, label = labels[1], fill = False, edgecolor = 'orange')
axes[2, 1].set(yscale = 'log', title = 'y position')
axes[2, 1].legend()
axes[2, 2].hist(dan_alpha, bins = 20, range = (0, np.max(dan_alpha)), label = labels[0], fill = True, edgecolor = 'blue')
axes[2, 2].hist(em_alpha, bins = 20, range = (0, np.max(dan_alpha)), label = labels[1], fill = False, edgecolor = 'orange')
axes[2, 2].set(yscale = 'log', title = 'alpha_Rs')
axes[2, 2].legend()
plt.savefig('plots/realization_histograms.png')

##############################################################

f_dan = h5py.File('cdm_kwargs_data.hdf5', 'r')
#f_dan = h5py.File('dan_kwargs_data.hdf5', 'r')
dan_alpha = f_dan['alpha_Rs'][:]
dan_Rs = f_dan['Rs'][:]
dan_x = f_dan['center_x'][:]
dan_y = f_dan['center_y'][:]
dan_rt = f_dan['r_trunc'][:]
f_dan.close()

f_bad = h5py.File('bad_em_kwargs_data.hdf5', 'r')
bad_alpha = f_bad['alpha_Rs'][:]
bad_Rs = f_bad['Rs'][:]
bad_x = f_bad['center_x'][:]
bad_y = f_bad['center_y'][:]
bad_rt = f_bad['r_trunc'][:]
f_bad.close()

f, axes = plt.subplots(5)
f.set_size_inches(15, 20)

axes[0].hist(dan_alpha, bins = 20, range = (0, 0.025), label = labels[0], fill = True, edgecolor = 'blue')
axes[0].hist(bad_alpha, bins = 20, range = (0, 0.025), label = labels[1], fill = False, edgecolor = 'orange')
axes[0].set(yscale = 'log', title = 'alpha_Rs')
axes[0].legend()
axes[1].hist(dan_Rs, bins = 20, range = (0, 8), label = labels[0], fill = True, edgecolor = 'blue')
axes[1].hist(bad_Rs, bins = 20, range = (0, 8), label = labels[1], fill = False, edgecolor = 'orange')
axes[1].set(yscale = 'log', title = 'Rs')
axes[1].legend()
axes[2].hist(dan_x, bins = 20, label = labels[0], fill = True, edgecolor = 'blue')
axes[2].hist(bad_x, bins = 20, label = labels[1], fill = False, edgecolor = 'orange')
axes[2].set(yscale = 'log', title = 'center_x')
axes[2].legend()
axes[3].hist(dan_y, bins = 20, label = labels[0], fill = True, edgecolor = 'blue')
axes[3].hist(bad_y, bins = 20, label = labels[1], fill = False, edgecolor = 'orange')
axes[3].set(yscale = 'log', title = 'center_y')
axes[3].legend()
axes[4].hist(dan_rt, bins = 20, range = (0, 25), label = labels[0], fill = True, edgecolor = 'blue')
axes[4].hist(bad_rt, bins = 20, range = (0, 25), label = labels[1], fill = False, edgecolor = 'orange')
axes[4].set(yscale = 'log', title = 'r_trunc')
axes[4].legend()
plt.savefig('plots/kwargs_histograms.png')

print('plots complete!')
