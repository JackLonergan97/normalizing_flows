import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import tensorflow_probability as tfp
import h5py
from samana.forward_model import forward_model
import os
import sys
import matplotlib.pyplot as plt
import time
import random
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

dm_model = sys.argv[1]

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
        #print('lines[' + str(i) + ']: ', lines[i])
    else:
        lines[i] = float(lines[i])
        #print('lines[' + str(i) + ']: ', lines[i])

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
emulator.load_weights('../data/emulatorModel' + dm_model)

def emulator_data(emulator = emulator, lines = lines, num_iterations = 1):

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
    i = 0 # used to count number of iterations
    n = 0
    reg_massInfall = []
    reg_massBound = []
    reg_concentration = []
    reg_redshift = []
    reg_orbitalRadius = []
    reg_projectedRadius = []
    reg_x = []
    reg_y = []
    reg_truncationRadius = []
    sample_amount = 1
    z = np.arange(stats.nbinom.ppf(0, r, p),stats.nbinom.ppf(0.9999999999999999, r, p))
    prob = stats.nbinom.pmf(z,r,p)
    prob = np.nan_to_num(prob)
    N = np.random.choice(z, p = prob)
    data = np.array([])

    min_concentration =  3.4845380492242852 # minimum concentration value from Galacticus data
    while n < num_iterations:
        samples = emulator.distribution.sample(sample_amount*N)
        x, _ = emulator.predict(samples, batch_size=65336)
        xt = norm_transform_inv(x, data_min, data_max, -1, 1)
        clip = (xt[:,0] > np.log10(2.0*massResolution/massTree)) & (xt[:,0] < np.log10(1e9/massHost)) & (xt[:,2] <= 0.0) & (xt[:,2] > -xt[:,0]+np.log10(massResolution/massTree)) & (xt[:,3] >= 0.0) & (xt[:,3] >= 0.34) & (xt[:,1] >= min_concentration)

        print('N: ', N)
        print('np.shape(data) before if/else statements: ', np.shape(data))
        print('len(data) before if/else statements: ', len(data))
        print('data.ndim before if/else statements: ', data.ndim)
        print('np.shape(xt): ', np.shape(xt))
        print('np.shape(xt[clip]): ', np.shape(xt[clip]))
        print('xt[clip].ndim: ', xt[clip].ndim)
        print('starting if/else statements')
        print('')

        if len(data) > N:
            data = data[:int(N)]
        elif len(data) < N:
            sample_amount += 1
            i += 1

            for j in range(len(xt[clip])):
                if len(data) == 0:
                    data = np.array([xt[clip][0]])
                else:
                    data = np.vstack((data, xt[clip][j]))
            print('np.shape(data) after iteration ' + str(i) + ':', np.shape(data))
            print('starting new iteration')
            print('')
            continue
        else:
            pass
        print('finished with if/else statements')
        print('data after if/else statements: ', data)
        print('np.shape(data): ', np.shape(data))
        print('')
        if isinstance(data[0], float):
            print('sampled N.B. value: ', N)
            print('exception data: ', data)
            print('shape of exception data: ', np.shape(data))
            print('')
            reg_massInfall.append(massHost * (10**data[0]))
            reg_massBound = [np.array(reg_massInfall[0] * 10**data[2])]
            reg_concentration.append(data[1])
            reg_redshift.append(data[3])
            reg_orbitalRadius.append(radiusVirialHost * (10**data[4]))
            reg_truncationRadius.append(radiusVirialHost * (10**data[5]))
            reg_projectedRadius.append(radiusVirialHost * (10**data[-1]))
            r2d_Mpc = radiusVirialHost * (10**data[-1])
            x = [0]*len(data)
            y = [0]*len(data)
        else:
            reg_massInfall.append(massHost * (10**data[:,0]))

            for i in range(len(data)): # We're doing this because the ith bound mass depends on the ith infall mass, whereas every other quantity depends on a  single scalar
                reg_massBound.append(reg_massInfall[0][i] * (10**data[i][2]))
            reg_massBound = [np.array(reg_massBound)]

            reg_concentration.append(data[:,1])
            reg_redshift.append(data[:,3])
            reg_orbitalRadius.append(radiusVirialHost * (10**data[:,4]))
            reg_truncationRadius.append(radiusVirialHost * (10**data[:,5]))
            reg_projectedRadius.append(radiusVirialHost * (10**data[:,-1]))
            r2d_Mpc = radiusVirialHost * (10**data[:,-1])
            x = [0]*len(data)
            y = [0]*len(data)

        for i in range(len(data)):
            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)

            theta = np.arccos(1 - 2*r1) # [0,pi] variable
            phi = 2 * np.pi * r2 # [0,2pi] variable

            x[i] = r2d_Mpc[i] * np.cos(phi)
            y[i] = r2d_Mpc[i] * np.sin(phi)
        reg_x.append(x)
        reg_y.append(y)
        n += 1

    return reg_massInfall, reg_concentration, reg_massBound, reg_redshift, reg_orbitalRadius, reg_truncationRadius, np.array(reg_x), np.array(reg_y)

#test = emulator_data()
#f = h5py.File('em_y_data.hdf5', 'w')
#f.create_dataset('r2d', data = test[-1])
#f.close()
#jkl

num_iterations = 1

# Constructing initialized parameters
output_path = os.getcwd() + '/emulator_inference_output_' + dm_model + '/' #CHANGE THIS WHEN WORKING ON STANDARD OR EMULATOR
#output_path = os.getcwd() + '/em_test/'
job_index = sys.argv[2]
#job_index = 1
n_keep = 100
#n_keep = 2
summary_statistic_tolerance = 1e5

from samana.Data.b1422 import B1422_HST
from samana.Model.b1422_model import B1422ModelEPLM3M4Shear
data_class = B1422_HST()
model = B1422ModelEPLM3M4Shear
preset_model_name = 'WDMEmulator'

kwargs_sample_realization = {}
kwargs_sample_realization['LOS_normalization'] = ['FIXED', 0.]
kwargs_sample_realization['log_m_host'] = ['FIXED', 13.3]
kwargs_sample_realization['cone_opening_angle_arcsec'] = ['FIXED', 8.0]
#kwargs_sample_realization['sigma_sub'] = ['FIXED', 3/5 * 0.357]
kwargs_sample_realization['sigma_sub'] = ['FIXED', 0.006]
kwargs_sample_realization['log_mlow'] = ['FIXED', 8.0]
kwargs_sample_realization['log_mhigh'] = ['FIXED', 9.0]

# WDM specific parameter
kwargs_sample_realization['log_mc'] = ['UNIFORM', 4.8, 10.0] 

# parameter for emulator data (keep commented out when not working with emulator)
kwargs_sample_realization['emulator_input'] = ['FIXED', emulator_data]

kwargs_sample_source = {'source_size_pc': ['UNIFORM', 25, 60]}
kwargs_sample_macro_fixed = {
    'a4_a': ['GAUSSIAN', 0.0, 0.01], 
    'a3_a': ['GAUSSIAN', 0.0, 0.005],
    'delta_phi_m3': ['GAUSSIAN', -np.pi/6, np.pi/6]
}
kwargs_model_class = {'shapelets_order': 10} # source complexity

# Run the simulation
forward_model(output_path, job_index, n_keep, data_class, model, preset_model_name, 
              kwargs_sample_realization, kwargs_sample_source, kwargs_sample_macro_fixed,
              tolerance=summary_statistic_tolerance, log_mlow_mass_sheets = 8.0, kwargs_model_class = kwargs_model_class, verbose=False, test_mode=False)

f = open(output_path + 'job_'+str(job_index)+'/parameters.txt', 'r')
param_names = f.readlines()[0]
print('PARAMETER NAMES:')
print(param_names)
f.close()

print('code executed!')
