import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from pyHalo.preset_models import CDM
from quadmodel.inference.forward_model import forward_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import h5py
import os
from scipy import stats
import time
import sys
import random
import warnings
warnings.filterwarnings('ignore')

np.random.seed(111)
countTree = 10

bound_mass = np.array([])
infall_mass = np.array([])
concentration = np.array([])
projected_radius = np.array([])
orbital_radius = np.array([])
infall_redshift = np.array([])
truncation_radius = np.array([])

num_subhalos = np.array([])
countSubhalosMean = 0
countSubhalosVariance = 0

# Generating countTree CDM realizations worth of subhalos
z_lens = 0.34
z_source = 3.62
kpc_per_arcsec = 5.013831007195271 # got this from lens_cosmo.cosmo.kpc_proper_per_asec(z_lens = 0.34) in pyHalo/PresetModels/external.py
kwargs = {
    'LOS_normalization' : 0.0,
    'log_m_host' : 13.3,
    'cone_opening_angle_arcsec' : 8.0,
    'sigma_sub' : 0.21419999999999997,
    'log_mlow' : 8.0,
    'shmf_log_slope' : -1.96
    }

for i in range(countTree): 
    cdm_realization = CDM(z_lens, z_source, **kwargs)
    countSubhalosMean += len(cdm_realization.halos)
    num_subhalos = np.append(num_subhalos, len(cdm_realization.halos))
    for j in range(len(cdm_realization.halos)):
        if cdm_realization.halos[j].is_subhalo == True:
            bound_mass = np.append(bound_mass, cdm_realization.halos[j].bound_mass)
            infall_mass = np.append(infall_mass, cdm_realization.halos[j].mass)
            concentration = np.append(concentration, cdm_realization.halos[j].c)
            orbital_radius = np.append(orbital_radius, cdm_realization.halos[j].r3d)
            infall_redshift = np.append(infall_redshift, cdm_realization.halos[j].z_infall)
            truncation_radius = np.append(truncation_radius, cdm_realization.halos[j].params_physical['r_trunc_kpc'])

            x_kpc = cdm_realization.halos[j].x * kpc_per_arcsec
            y_kpc = cdm_realization.halos[j].y * kpc_per_arcsec

            projected_radius = np.append(projected_radius, np.sqrt(x_kpc**2 + y_kpc**2))

        else:
            print('in realization ' + str(i) + ', halo ' + str(j) + ' is not a subhalo')
            pass
countSubhalosMean = countSubhalosMean/countTree

for i in range(countTree):
    countSubhalosVariance += (num_subhalos[i] - countSubhalosMean)**2/(countTree - 1)

# Normalizing these parameters into a 6D data set
bound_mass_normalized = np.log10(bound_mass/np.max(bound_mass))
infall_mass_normalized = np.log10(infall_mass/np.max(infall_mass))
concentration_normalized = np.log10(concentration/np.max(concentration))
projected_radius_normalized = np.log10(projected_radius/np.max(projected_radius))
orbital_radius_normalized = np.log10(orbital_radius/np.max(orbital_radius))
infall_redshift_normalized = np.log10(infall_redshift/np.max(infall_redshift))
truncation_radius_normalized = np.log10(truncation_radius/np.max(truncation_radius))


data=np.array(
    list(
        zip(
            infall_mass_normalized,
            bound_mass_normalized,
            concentration_normalized,
            projected_radius_normalized,
            orbital_radius_normalized,
            infall_redshift_normalized,
            truncation_radius_normalized
        )
    )
)

min_array = np.nanmin(data, axis = 0)
max_array = np.nanmax(data, axis = 0)
standard_max_data = np.array([np.max(infall_mass), np.max(bound_mass), np.max(concentration), np.max(projected_radius), np.max(orbital_radius), np.max(infall_redshift), np.max(truncation_radius)])
print('standard_max_data is: ')
print(standard_max_data)

def norm_transform(data, min_val, max_val):
    data_min = np.nanmin(data, axis = 0)
    data_max = np.nanmax(data, axis = 0)
    sigma_data = (data - data_min)/(data_max - data_min)
    return data_min, data_max, sigma_data*(max_val - min_val) + min_val

def norm_transform_inv(norm_data, data_min, data_max, min_val, max_val):
    data_min = np.nanmin(data, axis = 0)
    data_max = np.nanmax(data, axis = 0)
    sigma_data = (norm_data - min_val)/(max_val - min_val)
    return sigma_data*(data_max - data_min) + data_min

data_min, data_max, normalized_data = norm_transform(data,-1,1)
augmented_normalized_data = np.hstack((normalized_data, np.expand_dims(bound_mass,1)))
np.random.shuffle(augmented_normalized_data)
print("Number of subhalos: "+str(len(data)))

with open('necessary_dan_data.txt', 'w', newline='') as file:
    file.write(str(min_array).replace('\n','').replace(' ',' ') + '\n')
    file.write(str(max_array).replace('\n','').replace(' ',' ') + '\n')
    file.write(str(standard_max_data).replace('\n','').replace(' ',' ') + '\n')
    file.write(str(countSubhalosMean) + '\n')
    file.close()

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

model = RealNVP(num_coupling_layers=12)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

history = model.fit(
    augmented_normalized_data, batch_size=256, epochs=200, verbose=2, validation_split=0.2
)

model.save_weights('../data/danModel')

emulator = RealNVP(num_coupling_layers=12)
emulator.load_weights('../data/danModel')

# From Galacticus space to Gaussian space.
z, _ = emulator(normalized_data)

# From Gaussian space to Galacticus space.
samples = emulator.distribution.sample(len(infall_mass_normalized))
x, _ = emulator.predict(samples)
xt = norm_transform_inv(x, np.nanmin(x, axis = 0), np.nanmax(x, axis = 0), -1, 1)
clip = (xt[:,0] > -2) & (xt[:,0] < 0) & (xt[:,1] < 0) & (xt[:,3] < 0) & (xt[:,4] < 0) & (xt[:,5] > -1.28) & (xt[:,5] < 0) 

# Testing now to create Density plots
from scipy.stats import gaussian_kde

mass_bound_density_dan = np.vstack([data[:,0], data[:,1]])
z1_dan = gaussian_kde(mass_bound_density_dan)(mass_bound_density_dan)
mass_bound_density_generated = np.vstack([xt[:,0][clip], xt[:, 1][clip]])
z1_generated = gaussian_kde(mass_bound_density_generated)(mass_bound_density_generated)

concentration_density_dan = np.vstack([data[:,0], data[:,2]])
z2_dan = gaussian_kde(concentration_density_dan)(concentration_density_dan)
concentration_density_generated = np.vstack([xt[:,0][clip], xt[:,2][clip]])
z2_generated = gaussian_kde(concentration_density_generated)(concentration_density_generated)

orbital_radius_density_dan = np.vstack([data[:,0], data[:,3]])
z3_dan = gaussian_kde(orbital_radius_density_dan)(orbital_radius_density_dan)
orbital_radius_density_generated = np.vstack([xt[:,0][clip], xt[:,3][clip]])
z3_generated = gaussian_kde(orbital_radius_density_generated)(orbital_radius_density_generated)

projected_radius_density_dan = np.vstack([data[:,0], data[:,4]])
z4_dan = gaussian_kde(projected_radius_density_dan)(projected_radius_density_dan)
projected_radius_density_generated = np.vstack([xt[:,0][clip], xt[:,4][clip]])
z4_generated = gaussian_kde(projected_radius_density_generated)(projected_radius_density_generated)

infall_redshift_density_dan = np.vstack([data[:,0], data[:,5]])
z5_dan = gaussian_kde(infall_redshift_density_dan)(infall_redshift_density_dan)
infall_redshift_density_generated = np.vstack([xt[:,0][clip], xt[:,5][clip]])
z5_generated = gaussian_kde(infall_redshift_density_generated)(infall_redshift_density_generated)

truncation_radius_density_dan = np.vstack([data[:,0], data[:,6]])
z6_dan = gaussian_kde(truncation_radius_density_dan)(truncation_radius_density_dan)
truncation_radius_density_generated = np.vstack([xt[:,0][clip], xt[:,6][clip]])
z6_generated = gaussian_kde(truncation_radius_density_generated)(truncation_radius_density_generated)

f, axes = plt.subplots(6, 2)
f.set_size_inches(15, 18)

points = np.linspace(-2,0) # for y = x line

axes[0, 0].scatter(data[:,0], data[:,1], c = z1_dan, s=9)
axes[0, 0].plot(points, points, 'r-')
axes[0, 0].set(title="Daniel's Model", ylabel="bound mass")
axes[0, 0].set_ylim([-4, 0])
axes[0, 1].scatter(xt[:,0][clip], xt[:,1][clip], c = z1_generated, s=9)
axes[0, 1].set(title="Emulator", ylabel="bound mass")
axes[0, 1].set_ylim([-4, 0])
axes[1, 0].scatter(data[:,0], data[:,2], c = z2_dan, s=9)
axes[1, 0].set(ylabel="concentration")
axes[1, 0].set_ylim([-1.2, 0])
axes[1, 1].scatter(xt[:,0][clip], xt[:,2][clip], c = z2_generated, s=9)
axes[1, 1].set(ylabel="concentration")
axes[1, 1].set_ylim([-1.2, 0])
axes[2, 0].scatter(data[:,0], data[:,3], c = z3_dan, s=9)
axes[2, 0].set(ylabel="orbital radius")
axes[2, 0].set_ylim([-2, 0])
axes[2, 1].scatter(xt[:,0][clip], xt[:,3][clip], c = z3_generated, s=9)
axes[2, 1].set(ylabel="orbital radius")
axes[2, 1].set_ylim([-2, 0])
axes[3, 0].scatter(data[:,0], data[:,4], c = z4_dan, s=9)
axes[3, 0].set(ylabel="projected radius")
axes[3, 0].set_ylim([-2, 0])
axes[3, 1].scatter(xt[:,0][clip], xt[:,4][clip], c = z4_generated, s=9)
axes[3, 1].set(ylabel="projected radius")
axes[3, 1].set_ylim([-2, 0])
axes[4, 0].scatter(data[:,0], data[:,5], c = z5_dan, s=9)
axes[4, 0].set(ylabel="infall redshift")
axes[4, 0].set_ylim([-1.5, 0])
axes[4, 1].scatter(xt[:,0][clip], xt[:,5][clip], c = z5_generated, s=9)
axes[4, 1].set(ylabel="infall redshift")
axes[4, 1].set_ylim([-1.5, 0])
axes[5, 0].scatter(data[:,0], data[:,6], c = z6_dan, s=9)
axes[5, 0].set(xlabel="infall mass", ylabel="truncation radius")
axes[5, 0].set_ylim([-2.0, 0])
axes[5, 1].scatter(xt[:,0][clip], xt[:,6][clip], c = z6_generated, s=9)
axes[5, 1].set(xlabel="infall mass", ylabel="truncation radius")
axes[5, 1].set_ylim([-2.0, 0])
plt.savefig('plots/dan_density.png')

f, axes = plt.subplots(7)
f.set_size_inches(15, 20)

axes[0].hist(data[:,0], bins = 70, label = "Daniel's Model", fill = True, edgecolor = 'blue')
axes[0].hist(xt[:,0][clip], bins = 70, label = 'Emulator', fill = False, edgecolor = 'orange')
axes[0].set(yscale = 'log', title = 'infall mass')
axes[0].legend()
axes[1].hist(data[:,1], bins = 70, label = "Daniel's Model", fill = True, edgecolor = 'blue')
axes[1].hist(xt[:,1][clip], bins = 70, label = 'Emulator', fill = False, edgecolor = 'orange')
axes[1].set(yscale = 'log', title = 'bound mass')
axes[1].legend()
axes[2].hist(data[:,2], bins = 70, label = "Daniel's Model", fill = True, edgecolor = 'blue')
axes[2].hist(xt[:,2][clip], bins = 70, label = 'Emulator', fill = False, edgecolor = 'orange')
axes[2].set(yscale = 'log', title = 'concentration')
axes[2].legend()
axes[3].hist(data[:,3], bins = 70, label = "Daniel's Model", fill = True, edgecolor = 'blue')
axes[3].hist(xt[:,3][clip], bins = 70, label = 'Emulator', fill = False, edgecolor = 'orange')
axes[3].set(yscale = 'log', title = 'orbital radius')
axes[3].legend()
axes[4].hist(data[:,4], bins = 70, label = "Daniel's Model", fill = True, edgecolor = 'blue')
axes[4].hist(xt[:,4][clip], bins = 70, label = 'Emulator', fill = False, edgecolor = 'orange')
axes[4].set(yscale = 'log', title = 'projected radius')
axes[4].legend()
axes[5].hist(data[:,5], bins = 70, label = "Daniel's Model", fill = True, edgecolor = 'blue')
axes[5].hist(xt[:,5][clip], bins = 70, label = 'Emulator', fill = False, edgecolor = 'orange')
axes[5].set(yscale = 'log', title = 'infall redshift')
axes[5].legend()
axes[6].hist(data[:,6], bins = 70, label = "Daniel's Model", fill = True, edgecolor = 'blue')
axes[6].hist(xt[:,6][clip], bins = 70, label = 'Emulator', fill = False, edgecolor = 'orange')
axes[6].set(yscale = 'log', title = 'truncation radius')
axes[6].legend()
plt.savefig('plots/dan_histograms.png')

print('code executed!')
