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

standard_CDM_output_path = os.getcwd() + '/standard_inference_output_CDM_res' + resolution + '_annulus/'
standard_WDM_output_path = os.getcwd() + '/standard_inference_output_WDM_res' + resolution + '_annulus/'
emulator_CDM_output_path = os.getcwd() + '/emulator_inference_output_CDM_res' + resolution + '_annulus/'
emulator_WDM_output_path = os.getcwd() + '/emulator_inference_output_WDM_res' + resolution + '_annulus/'
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
#    stat = np.loadtxt(emulator_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,2]
    try:
        stat = np.loadtxt(standard_CDM_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,2]
        standard_CDM_statistic = np.append(standard_CDM_statistic, stat)

        #if len(stat) < 100:
        #    print('CDM job_' + str(i) + ' only has ' + str(len(stat)) + ' simulation outputs') 

        stat = np.loadtxt(standard_WDM_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,3]
        standard_WDM_statistic = np.append(standard_WDM_statistic, stat)

        stat = np.loadtxt(emulator_CDM_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,2]
        emulator_CDM_statistic = np.append(emulator_CDM_statistic, stat)

        #if len(stat) < 100:
        #    print('emulator job_' + str(i) + ' only has ' + str(len(stat)) + ' simulation outputs')

        stat = np.loadtxt(emulator_WDM_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,3]
        emulator_WDM_statistic = np.append(emulator_WDM_statistic, stat)
    
        #stat = np.loadtxt(dan_CDM_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,-2]
        #dan_CDM_statistic = np.append(dan_CDM_statistic, stat)
    except:
        print('incorrect path name used!')
        pass

fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.set_size_inches(17, 5)

hist, bins = np.histogram(standard_CDM_statistic, bins = 20)
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
print('CDM min, max bins: ', np.log10(bins[0]), np.log10(bins[-1]))
ax1.hist(standard_CDM_statistic, bins = logbins, label = 'CDM', fill = False, edgecolor = 'blue')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('$S_{lens}$')
ax1.hist(emulator_CDM_statistic, bins = logbins, label = 'Emulator', fill = False, edgecolor = 'orange')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('$S_{lens}$')
ax1.legend()

hist, bins = np.histogram(emulator_CDM_statistic, bins = 20)
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
print('DMEmulator min, max bins: ', np.log10(bins[0]), np.log10(bins[-1]))
ax2.hist(standard_CDM_statistic, bins = logbins, label = 'CDM', fill = False, edgecolor = 'blue')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('$S_{lens}$')
ax2.hist(emulator_CDM_statistic, bins = logbins, label = 'Emulator', fill = False, edgecolor = 'orange')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('$S_{lens}$')
ax2.legend()
plt.savefig('plots/hist_stat.png')

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

    n = np.sum(standard_WDM_statistic > xi)
    x_o_WDM.append(n)

    n = np.sum(emulator_CDM_statistic > xi)
    x_e_CDM.append(n)

    n = np.sum(emulator_WDM_statistic > xi)
    x_e_WDM.append(n)

    #n = np.sum(dan_CDM_statistic > xi)
    #x_d_CDM.append(n)

x_o_CDM = np.array(x_o_CDM)/len(standard_CDM_statistic)
x_o_WDM = np.array(x_o_WDM)/len(standard_WDM_statistic)
x_e_CDM = np.array(x_e_CDM)/len(emulator_CDM_statistic)
x_e_WDM = np.array(x_e_WDM)/len(emulator_WDM_statistic)
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

    n = np.sum(standard_WDM_statistic < xi)
    x_o_WDM.append(n)

    n = np.sum(emulator_CDM_statistic < xi)
    x_e_CDM.append(n)

    n = np.sum(emulator_WDM_statistic < xi)
    x_e_WDM.append(n)

    #n = np.sum(dan_CDM_statistic < xi)
    #x_d_CDM.append(n)

x_o_CDM = np.array(x_o_CDM)/len(standard_CDM_statistic)
#print('length of x_o_CDM: ' + str(len(x_o_CDM)))
x_o_WDM = np.array(x_o_WDM)/len(standard_WDM_statistic)
#print('length of x_o_WDM: ' + str(len(x_o_WDM)))
x_e_CDM = np.array(x_e_CDM)/len(emulator_CDM_statistic)
#print('length of x_e_CDM: ' + str(len(x_e_CDM)))
x_e_WDM = np.array(x_e_WDM)/len(emulator_WDM_statistic)
#print('length of x_e_WDM: ' + str(len(x_e_WDM)))
#x_d_CDM = np.array(x_d_CDM)/len(dan_CDM_statistic)

fig = plt.figure()
plt.plot(s, x_o_WDM/x_o_CDM, label = 'Original')
plt.plot(s, x_e_WDM/x_e_CDM, label = 'Emulator')
plt.xlim((0, 0.1))
plt.xlabel('$S_{lens}$')
plt.ylabel('WDM to CDM Ratio')
plt.legend()
plt.savefig('plots/likelihood_ratio_res' + resolution + '.png')

fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.set_size_inches(17, 5)

ax1.plot(s, x_o_CDM, 'k-', label = 'CDM')
ax1.plot(s, x_e_CDM, 'r-', label = 'CDMEmulator')
ax1.plot(s, x_o_WDM, 'k-.', label = 'WDM')
ax1.plot(s, x_e_WDM, 'r-.', label = 'WDMEmulator')
ax1.set_xlim([0, 2])
ax1.set_xlabel('$S_{lens}$')
ax1.set_ylabel('Fraction < $S_{lens}$')
ax1.legend()
ax2.plot(s, x_o_CDM, 'k-', label = 'CDM')
ax2.plot(s, x_e_CDM, 'r-', label = 'CDMEmulator')
ax2.plot(s, x_o_WDM, 'k-.', label = 'WDM')
ax2.plot(s, x_e_WDM, 'r-.', label = 'WDMEmulator')
ax2.set_xlim([0, 0.1])
ax2.set_ylim([0, 0.1])
ax2.set_xlabel('$S_{lens}$')
ax2.set_ylabel('Fraction < $S_{lens}$')
ax2.legend()
plt.savefig('plots/inverted_dm_ssd_res' + resolution + '.png')

print('plots complete!')
