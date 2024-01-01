import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

resolution = sys.argv[1]
n_files = 100

# Create the summary statistic distribution plot
#for model in dm_models:
#    emulator_output_path = os.getcwd() + '/emulator_inference_output_' + model + '/'
#    standard_output_path = os.getcwd() + '/standard_inference_output_CDM/'
    
#    emulator_statistic = np.array([])
#    standard_statistic = np.array([])

standard_CDM_output_path = os.getcwd() + '/standard_inference_output_CDM/'
standard_WDM_output_path = os.getcwd() + '/standard_inference_output_WDM/'
emulator_CDM_output_path = os.getcwd() + '/emulator_inference_output_CDM_res' + resolution + '/'
emulator_WDM_output_path = os.getcwd() + '/emulator_inference_output_WDM_res' + resolution + '/'

standard_CDM_statistic = np.array([])
standard_WDM_statistic = np.array([])
emulator_CDM_statistic = np.array([])
emulator_WDM_statistic = np.array([])

for i in range(1, n_files+1):
#    stat = np.loadtxt(emulator_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,-2]
    try:
        stat = np.loadtxt(standard_CDM_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,-2]
        standard_CDM_statistic = np.append(standard_CDM_statistic, stat)

        stat = np.loadtxt(standard_WDM_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,-2]
        standard_WDM_statistic = np.append(standard_WDM_statistic, stat)

        stat = np.loadtxt(emulator_CDM_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,-2]
        emulator_CDM_statistic = np.append(emulator_CDM_statistic, stat)

        stat = np.loadtxt(emulator_WDM_output_path + 'job_' + str(i) + '/parameters.txt', skiprows = 1)[:,-2]
        emulator_WDM_statistic = np.append(emulator_WDM_statistic, stat)
    except:
        print(i)
        pass

print('standard_CDM_statistic')
print(standard_CDM_statistic)
print('standard_WDM_statistic')
print(standard_WDM_statistic)
print('emulator_CDM_statistic')
print(emulator_CDM_statistic)
print('emulator_WDM_statistic')
print(emulator_WDM_statistic)

# Making the downward curved plot
s = np.linspace(0, 10., 10000)
x_o_CDM = []
x_o_WDM = []
x_e_CDM = []
x_e_WDM = []

for xi in s:
    n = np.sum(standard_CDM_statistic > xi)
    x_o_CDM.append(n)

    n = np.sum(standard_WDM_statistic > xi)
    x_o_WDM.append(n)

    n = np.sum(emulator_CDM_statistic > xi)
    x_e_CDM.append(n)

    n = np.sum(emulator_WDM_statistic > xi)
    x_e_WDM.append(n)

x_o_CDM = np.array(x_o_CDM)/len(standard_CDM_statistic)
x_o_WDM = np.array(x_o_WDM)/len(standard_WDM_statistic)
x_e_CDM = np.array(x_e_CDM)/len(emulator_CDM_statistic)
x_e_WDM = np.array(x_e_WDM)/len(emulator_WDM_statistic)

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
for xi in s:
    n = np.sum(standard_CDM_statistic < xi)
    x_o_CDM.append(n)

    n = np.sum(standard_WDM_statistic < xi)
    x_o_WDM.append(n)

    n = np.sum(emulator_CDM_statistic < xi)
    x_e_CDM.append(n)

    n = np.sum(emulator_WDM_statistic < xi)
    x_e_WDM.append(n)

print('x_o_CDM')
print(x_o_CDM)
print('x_o_WDM')
print(x_o_WDM)
print('x_e_CDM')
print(x_e_CDM)
print('x_e_WDM')
print(x_e_WDM)

x_o_CDM = np.array(x_o_CDM)/len(standard_CDM_statistic)
print('length of x_o_CDM: ' + str(len(x_o_CDM)))
x_o_WDM = np.array(x_o_WDM)/len(standard_WDM_statistic)
print('length of x_o_WDM: ' + str(len(x_o_WDM)))
x_e_CDM = np.array(x_e_CDM)/len(emulator_CDM_statistic)
print('length of x_e_CDM: ' + str(len(x_e_CDM)))
x_e_WDM = np.array(x_e_WDM)/len(emulator_WDM_statistic)
print('length of x_e_WDM: ' + str(len(x_e_WDM)))

#plt.plot(s, x_o_WDM/x_o_CDM)
#plt.plot(s, x_e_WDM/x_e_CDM)
#plt.xlim((0, 0.1))
#plt.savefig('plots/likelihood_ratio')

fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.set_size_inches(17, 5)

ax1.plot(s, x_o_CDM, label = 'CDM')
ax1.plot(s, x_o_WDM, label = 'WDM')
ax1.plot(s, x_e_CDM, label = 'CDMEmulator')
ax1.plot(s, x_e_WDM, label = 'WDMEmulator')
ax1.set_xlim([0, 2])
ax1.set_xlabel('$S_{lens}$')
ax1.set_ylabel('Fraction < $S_{lens}$')
ax1.legend()
ax2.plot(s, x_o_CDM, label = 'CDM')
ax2.plot(s, x_o_WDM, label = 'WDM')
ax2.plot(s, x_e_CDM, label = 'CDMEmulator')
ax2.plot(s, x_e_WDM, label = 'WDMEmulator')
ax2.set_xlim([0, 0.1])
ax2.set_ylim([0, 0.1])
ax2.set_xlabel('$S_{lens}$')
ax2.set_ylabel('Fraction < $S_{lens}$')
ax2.legend()
plt.savefig('plots/inverted_dm_ssd_res' + resolution + '.png')

print('plots complete!')
