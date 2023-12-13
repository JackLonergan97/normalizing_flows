import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import tensorflow_probability as tfp
import h5py
from quadmodel.inference.forward_model import forward_model
import os
import sys
import matplotlib.pyplot as plt
import time
import random
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

dm_model = sys.argv[1]
#dm_model = 'CDM'

# Reading in the data and converting it from a string to a list of lists and floats.
necessary_data = open("necessary_data_" + dm_model + ".txt", "r")
lines = necessary_data.readlines()
necessary_data.close()

# removing the "\n" at the end of every line from the .txt file
for i in range(len(lines)):
    lines[i] = lines[i].strip('\n')
    if i < 2:
        lines[i] = lines[i][1:-2]
        lines[i].split()
        lines[i] = [float(i) for i in lines[i].split()]
    else:
        lines[i] = float(lines[i])

## removing the "\n" at the end of every line from the .txt file
#for i in range(len(lines)):
#    lines[i] = lines[i].strip('\n')
#    if i < 2:
#        lines[i] = lines[i][1:-2]
#        lines[i].split()
#        lines[i] = [float(i) for i in lines[i].split()]
#    else:
#        lines[i] = float(lines[i])
        
## Assigning variable names to each of the data components
##data_min = np.array(lines[0])
#data_max = np.array(lines[1])
#massTree = lines[2]
#massResolution = lines[3]
#massHost = lines[4]
#radiusVirialHost = lines[5]
#countSubhalosMean = lines[6]

#def norm_transform_inv(norm_data, min_val, max_val):
#     data_min = np.nanmin(data, axis = 0)
#     data_max = np.nanmax(data, axis = 0)
#    sigma_data = (norm_data - min_val)/(max_val - min_val)
#    return sigma_data*(data_max - data_min) + data_min

# put in conditions to prevent from crashing (min_val less than max_val/ sizes of data_min data_max)
def norm_transform_inv(norm_data, data_min, data_max, min_val = -1 , max_val = 1):
    sigma_data = (norm_data - min_val)/(max_val - min_val)
    return sigma_data*(data_max - data_min) + data_min

# Creating a custom layer with keras API.
output_dim = 256
reg = 0.01

def Coupling(input_shape):
    input = keras.layers.Input(shape=input_shape)

    t_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    t_layer_2 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_1)
    t_layer_3 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_2)
    t_layer_4 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_3)
    t_layer_5 = keras.layers.Dense(
        input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_4)

    s_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    s_layer_2 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_1)
    s_layer_3 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_2)
    s_layer_4 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_3)
    s_layer_5 = keras.layers.Dense(
        input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_4)

    return keras.Model(inputs=input, outputs=[s_layer_5, t_layer_5])

class RealNVP(keras.Model):
    def __init__(self, num_coupling_layers):
        super(RealNVP, self).__init__()

        self.num_coupling_layers = num_coupling_layers

        # Distribution of the latent space.
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], scale_diag=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        self.masks = np.array(
            [[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]] * (num_coupling_layers // 2), dtype="float32"
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.layers_list = [Coupling(6) for i in range(num_coupling_layers)]

    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def call(self, x, training=True):
        log_det_inv = 0
        direction = 1
        if training:
            direction = -1
        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
                + x_masked
            )
            log_det_inv += gate * tf.reduce_sum(s, [1])

        return x, log_det_inv

    # Log likelihood of the normal distribution plus the log determinant of the jacobian.

    def log_loss(self, data):
        # Extract the actual data here as "x", and the final weight column as "w".
        x = data[:,0:-1]
        w = data[:,-1]
        m = data[:,0]
        y, logdet = self(x)
        # Suppose the weight of the subhalo is "N". This means that this subhalo actually represents N such subhalos.
        # Treating these as independent contributions to the likelihood, we should multiply the probability, p, of this point
        # together N times, i.e. p^N. Since we compute a log-likelihood this corresponds to multiplying the likelihood by the weight.
        log_likelihood = (self.distribution.log_prob(y) + logdet)*w
        return -tf.reduce_mean(log_likelihood)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.log_loss(data)

        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

emulator = RealNVP(num_coupling_layers=12)
emulator.load_weights('../data/emulatorModel' + dm_model)

#def emulator_data():
#    # reading in necessary data so that N can be drawn from a N.B. distribution rather than N = 30,000
#    s_I = 0.18
#    p = 1/(1 + countSubhalosMean*s_I**2)
#    r = 1/s_I**2
#    x = np.arange(stats.nbinom.ppf(0, r, p),stats.nbinom.ppf(0.9999999999999999, r, p))
#
#    # From latent space to data.
#    N = np.random.choice(x, p = stats.nbinom.pmf(x,r,p))
#    samples = emulator.distribution.sample(1.5*N)
#    x, _ = emulator.predict(samples)
#    xt = norm_transform_inv(x, -1, 1)
#    print('xt is: ' + str(xt))
#    clip = (xt[:,0] > np.log10(2.0*massResolution/massTree)) & (xt[:,2] <= 0.0) & (xt[:,2] > -xt[:,0]+np.log10(massResolution/massTree)) & (xt[:,3] >= 0.0)     & (xt[:,1] > 0.0)
#    print('xt[clip] is: ' + str(xt[clip]))
#    data = xt[clip]
#
#    reg_MassInfall = massHost * (10**data[:,0])
#    reg_MassBound = reg_MassInfall * (10**data[:,2])
#   # r_kpc = 1000 * radiusVirialHost * (10**data[:,4])
#
#    massInfall = reg_MassInfall
#    massBound = reg_MassBound
#    redshifts = [0.5] * len(massInfall)
#    concentration = data[:,1]
#    halo_list = []
#
#    # Creating a set of x,y positions
#    x = np.zeros(len(data))
#    y = np.zeros(len(data))
#
#    for i in range(len(data)):
#        r1 = random.uniform(0, 1)
#        r2 = random.uniform(0, 1)
#
#        theta = np.arccos(1 - 2*r1) # [0,pi] variable
#        phi = 2 * np.pi * r2 # [0,2pi] variable
#
#        x[i] = r_kpc[i] * np.cos(phi) * np.sin(theta)
#        y[i] = r_kpc[i] * np.sin(phi) * np.sin(theta)
#
#    return [massInfall, x, y, massBound, concentration]

def emulator_data(emulator, lines, num_iterations):

    data_min = np.array(lines[0])
    data_max = np.array(lines[1])
    massTree = lines[2]
    massResolution = lines[3]
    massHost = lines[4]
    radiusVirialHost = lines[5]
    countSubhalosMean = lines[6]

    # reading in necessary data so that N can be drawn from a N.B. distribution rather than N = 30,000
    s_I = 0.18
    p = 1/(1 + countSubhalosMean*s_I**2)
    r = 1/s_I**2

    # initializing arrays to go from latent space to data
    n = 0
    reg_massInfall = []
    reg_massBound = []
    reg_concentration = []
    reg_redshift = []
    reg_orbitalRadius = []
    reg_x = []
    reg_y = []
    reg_truncationRadius = []
    sample_amount = 1.5
    z = np.arange(stats.nbinom.ppf(0, r, p),stats.nbinom.ppf(0.9999999999999999, r, p))
    prob = stats.nbinom.pmf(z,r,p)
    prob = np.nan_to_num(prob)

    while n < num_iterations:
        N = np.random.choice(z, p = prob)
        print('starting sampling emulator points')
        samples = emulator.distribution.sample(sample_amount*N)
        print('ending sampling emulator points')
        x, _ = emulator.predict(samples, batch_size=65336)
        print('x is ' + str(x[0:20]))
        xt = norm_transform_inv(x, data_min, data_max, -1, 1)
        clip = (xt[:,0] > np.log10(2.0*massResolution/massTree)) & (xt[:,2] <= 0.0) & (xt[:,2] > -xt[:,0]+np.log10(massResolution/massTree)) & (xt[:,3] >= 0.0)


        ts = time.time()
        time_format = time.strftime("%H:%M:%S", time.gmtime(ts))
        print('Starting to create subhalo population: ' + str(time_format))
        print(' ')
        if len(xt[clip]) > N:
            data = xt[clip][:int(N)]
        elif len(xt[clip]) < N:
            print('Not enough data points are being sampled in iteration ' + str(n + 1))
            sample_amount += 0.5
            continue
        else:
            data = xt[clip]
        ts = time.time()
        time_format = time.strftime("%H:%M:%S", time.gmtime(ts))
        print('Finished creating subhalo population: ' + str(time_format))
        print(' ')
        print('starting the append process')

        reg_massInfall.append(massHost * (10**data[:,0]))
        reg_massBound.append(reg_massInfall[n] * (10**data[:,2]))
        reg_concentration.append(data[:,1])
        reg_redshift.append(data[:,3])
        reg_orbitalRadius.append(radiusVirialHost * (10**data[:,4]))
        reg_truncationRadius.append(radiusVirialHost * (10**data[:,5]))
        r_kpc = 1000 * radiusVirialHost * (10**data[:,4])
        x = [0]*len(r_kpc)
        y = [0]*len(r_kpc)

        for i in range(len(data)):
            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)

            theta = np.arccos(1 - 2*r1) # [0,pi] variable
            phi = 2 * np.pi * r2 # [0,2pi] variable

            x[i] = r_kpc[i] * np.cos(phi) * np.sin(theta)
            y[i] = r_kpc[i] * np.sin(phi) * np.sin(theta)
        reg_x.append(x)
        reg_y.append(y)
        print('ending the append process')
        n += 1

    return [reg_massInfall, reg_concentration, reg_massBound, reg_redshift, reg_orbitalRadius, reg_truncationRadius, np.array(reg_x), np.array(reg_y)]

num_iterations = 1

# Constructing initialized parameters
output_path = os.getcwd() + '/emulator_inference_output_' + dm_model + '/'
#job_index = sys.argv[2]
job_index = 1
n_keep = 100
summary_statistic_tolerance = 1e5
lens_data = 'B1422'
from quadmodel.data.b1422 import B1422
lens_data = B1422()
print(lens_data.m)

realization_priors = {}
#realization_priors['PRESET_MODEL'] = 'DMEmulator'
realization_priors['PRESET_MODEL'] = 'CDM'
realization_priors['LOS_normalization'] = ['FIXED', 1.]
realization_priors['log_m_host'] = ['FIXED', 13.3]
realization_priors['cone_opening_angle_arcsec'] = ['FIXED', 8.0]

# Testing out things that are in the "example_summary_statistic_distribution.py" script
realization_priors['sigma_sub'] = ['FIXED', 0.05]
realization_priors['log_mlow'] = ['FIXED', 9.0]

# WDM Specific parameters (keep commented out when working with CDM)
#realization_priors['sigma_sub'] = ['UNIFORM', 0.0, 0.01]
#realization_priors['log_mc'] = ['UNIFORM', 4.8, 10.0]

# parameter for emulator data (only un-comment when working with CDMEmulator or WDMEmulator)
#realization_priors['emulator_input'] = ['FIXED', emulator_data(emulator, lines, num_iterations)]

macromodel_priors = {}
macromodel_priors['m4_amplitude_prior'] = [np.random.normal, 0.0, 0.01]
macromodel_priors['gamma_macro_prior'] = [np.random.uniform, 1.8, 2.3]
# FOR A CUSTOM SHEAR PRIOR:
macromodel_priors['shear_strength_prior'] = [np.random.uniform, 0.05, 0.25]

# the present lenses also have built-in shear priors determined based on what values get accepted after running ABC;
# using a broader prior, you will waste some time exploring parameter space that will get rejected
shear_min, shear_max = lens_data.kwargs_macromodel['shear_amplitude_min'], lens_data.kwargs_macromodel['shear_amplitude_max']
print(shear_min, shear_max)
macromodel_priors['shear_strength_prior'] = [np.random.uniform, shear_min, shear_max]

# Run the simulation
forward_model(output_path, job_index, lens_data, n_keep, realization_priors, macromodel_priors,
              tolerance=summary_statistic_tolerance, verbose=False, test_mode=False, save_realizations=True)

f = open(output_path + 'job_'+str(job_index)+'/parameters.txt', 'r')
param_names = f.readlines()[0]
print('PARAMETER NAMES:')
print(param_names)
f.close()
print('simulation finished!')

# accepted_parameters = np.loadtxt(output_path + 'job_'+str(job_index)+'/parameters.txt', skiprows=1)
# print('ACCEPTED PARAMETERS:')
# print(accepted_parameters)
# the first set of parameters are the ones specified in kwargs_realization (see cell #2), the rest are the source size,
# macromodel parameters, and the last parameter is the summary statistic

# accepeted_mags = np.loadtxt(output_path + 'job_'+str(job_index)+'/fluxes.txt')
# print('\nACCEPTED MAGNIFICATIONS:')
# print(accepeted_mags)
