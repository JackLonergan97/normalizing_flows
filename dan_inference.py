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
necessary_data = open("necessary_dan_data.txt", "r")
lines = necessary_data.readlines()
necessary_data.close()

# removing the "\n" at the end of every line from the .txt file
for i in range(len(lines)):
    lines[i] = lines[i].strip('\n')
    if i < 3:
        lines[i] = lines[i][1:-1]
        lines[i].split()
        lines[i] = [float(i) for i in lines[i].split()]
    else:
        lines[i] = float(lines[i])

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
            loc=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], scale_diag=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        self.masks = np.array(
            [[1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1]] * (num_coupling_layers // 2), dtype="float32"
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.layers_list = [Coupling(7) for i in range(num_coupling_layers)]

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
        m = data[:,0]
        y, logdet = self(x)
        # Suppose the weight of the subhalo is "N". This means that this subhalo actually represents N such subhalos.
        # Treating these as independent contributions to the likelihood, we should multiply the probability, p, of this point
        # together N times, i.e. p^N. Since we compute a log-likelihood this corresponds to multiplying the likelihood by the weight.
        log_likelihood = self.distribution.log_prob(y) + logdet
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
emulator.load_weights('../data/danModel')

def dan_emulator_data(emulator = emulator, lines = lines, num_iterations = 1):

    min_array = np.array(lines[0])
    max_array = np.array(lines[1])
    data_max = lines[2]
    countSubhalosMean = lines[3]

    # initializing arrays to go from latent space to data
    n = 0
    KPC_TO_MPC = 1e-3
    kpc_per_arcsec = 5.013831007195271 # at z_lens = 0.34
    reg_massInfall = []
    reg_massBound = []
    reg_concentration = []
    reg_redshift = []
    reg_projectedRadius = []
    reg_orbitalRadius = []
    reg_x = []
    reg_y = []
    reg_truncationRadius = []
    sample_amount = 1.5

    while n < num_iterations:
        N = stats.poisson.rvs(mu = countSubhalosMean)
        samples = emulator.distribution.sample(sample_amount*N)
        x, _ = emulator.predict(samples, batch_size=65336)
        xt = norm_transform_inv(x, min_array, max_array, -1, 1)
        clip = (xt[:,0] > -2) & (xt[:,0] < 0) & (xt[:,1] < 0) & (xt[:,3] < 0) & (xt[:,4] < 0) & (xt[:,5] > -1.28) & (xt[:,5] < 0)

        if len(xt[clip]) > N:
            data = xt[clip][:int(N)]
        elif len(xt[clip]) < N:
            print('Not enough data points are being sampled in iteration ' + str(n + 1))
            sample_amount += 0.5
            continue
        else:
            data = xt[clip]

        reg_massInfall.append(data_max[0] * (10**data[:,0]))
        reg_massBound.append(data_max[1] * (10**data[:,1]))
        reg_concentration.append(data_max[2] * (10**data[:,2]))
        reg_projectedRadius.append(KPC_TO_MPC * data_max[3] * (10**data[:,3]))
        reg_orbitalRadius.append(KPC_TO_MPC * data_max[4] * (10**data[:,4]))
        reg_redshift.append(data_max[5] * (10**data[:,5]))
        reg_truncationRadius.append(KPC_TO_MPC * data_max[6] * (10**data[:,6]))
        r2d_Mpc = KPC_TO_MPC * data_max[3] * 10**data[:,3]

        x = [0]*len(data[:,0])
        y = [0]*len(data[:,0])

        for i in range(len(data)):
            r = random.uniform(0, 1)

            theta = 2 * np.pi * r # [0,2pi] variable

            x[i] = r2d_Mpc[i] * np.cos(theta)
            y[i] = r2d_Mpc[i] * np.sin(theta)
        reg_x.append(x)
        reg_y.append(y)
        print('ending the append process')
        n += 1

    return reg_massInfall, reg_concentration, reg_massBound, reg_redshift, reg_orbitalRadius, reg_truncationRadius, np.array(reg_x), np.array(reg_y)

num_iterations = 1

# Constructing initialized parameters
output_path = os.getcwd() + '/em_test/'
job_index = sys.argv[2]
#job_index = 1
n_keep = 100
#n_keep = 1
summary_statistic_tolerance = 1e5
lens_data = 'B1422'
from quadmodel.data.b1422 import B1422
lens_data = B1422()

realization_priors = {}
realization_priors['PRESET_MODEL'] = 'DMEmulator'
realization_priors['LOS_normalization'] = ['FIXED', 0.]
realization_priors['log_m_host'] = ['FIXED', 13.3]
realization_priors['cone_opening_angle_arcsec'] = ['FIXED', 8.0]

# Testing out things that are in the "example_summary_statistic_distribution.py" script
realization_priors['sigma_sub'] = ['FIXED', 3/5 * 0.357]
realization_priors['log_mlow'] = ['FIXED', 8.0]
realization_priors['log_mhigh'] = ['FIXED', 10.0]

# WDM Specific parameters (keep commented out when working with CDM)
#realization_priors['sigma_sub'] = ['UNIFORM', 0.0, 0.01]
#realization_priors['log_mc'] = ['UNIFORM', 4.8, 10.0]

#f = h5py.File('dan_subhalo_data.hdf5', 'r')
#m_infall = f['m_infall'][:]
#c = f['c'][:]
#m_bound = f['m_bound'][:]
#z_infall = f['z_infall'][:]
#r3d = f['r3d'][:]
#rt = f['rt'][:]
#x = f['x'][:]
#y = f['y'][:]
#f.close()

#print('within dan_inference.py')
#print('number of subhalos: ', len(m_infall))
#print('infall mass is: ', m_infall[0:5])
#print('concentration is: ', c[0:5])
#print('bound mass is: ', m_bound[0:5])
#print('infall_redshift is: ', z_infall[0:5])
#print('orbital radius is: ', r3d[0:5])
#print('truncation radius is: ', rt[0:5])
#print('x position (arcseconds) is: ', x[0:5])
#print('y position (arcseconds) is: ', y[0:5])
#print('')

# USE FOR THE SINGLE SUBHALO TEST
#emulator_array = [m_infall, c, m_bound, z_infall, r3d, rt, x, y]

# USE WHEN WE HAVE A SINGLE POPULATION OF SUBHALOS TO PRODUCE ONE S_LENS VALUE
#emulator_array = [[m_infall], [c], [m_bound], [z_infall], [r3d], [rt], [x], [y]]

#realization_priors['emulator_input'] = ['FIXED', emulator_array]
realization_priors['emulator_input'] = ['FIXED', dan_emulator_data]

macromodel_priors = {}
macromodel_priors['m4_amplitude_prior'] = [np.random.normal, 0.0, 0.01]
#macromodel_priors['m4_amplitude_prior'] = ['FIXED', 0.005]

macromodel_priors['gamma_macro_prior'] = [np.random.uniform, 1.8, 2.3]
#macromodel_priors['gamma_macro_prior'] = ['FIXED', 2.0]

# the present lenses also have built-in shear priors determined based on what values get accepted after running ABC;
# using a broader prior, you will waste some time exploring parameter space that will get rejected
shear_min, shear_max = lens_data.kwargs_macromodel['shear_amplitude_min'], lens_data.kwargs_macromodel['shear_amplitude_max']
print(shear_min, shear_max)

macromodel_priors['shear_strength_prior'] = [np.random.uniform, shear_min, shear_max]
#macromodel_priors['shear_strength_prior'] = ['FIXED', 0.2]

# Run the simulation
forward_model(output_path, job_index, lens_data, n_keep, realization_priors, macromodel_priors,
              tolerance=summary_statistic_tolerance, verbose=False, test_mode=False, save_realizations=True)

f = open(output_path + 'job_'+str(job_index)+'/parameters.txt', 'r')
param_names = f.readlines()[0]
print('PARAMETER NAMES:')
print(param_names)
f.close()

print('code executed!')
