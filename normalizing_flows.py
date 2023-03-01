import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import h5py
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

f = h5py.File('darkMatterOnlySubHalos.hdf5', 'r')
mergerTreeBuildMassesGroup = f['Parameters/mergerTreeBuildMasses']
massResolutionGroup = f['Parameters/mergerTreeMassResolution']
massTree = mergerTreeBuildMassesGroup.attrs['massTree'][0]
countTree = mergerTreeBuildMassesGroup.attrs['treeCount'][0]
massResolution = massResolutionGroup.attrs['massResolution']
weight = f['Outputs/Output1/nodeData/nodeSubsamplingWeight']
treeIndex = f['Outputs/Output1/nodeData/mergerTreeIndex']
isCentral = f['Outputs/Output1/nodeData/nodeIsIsolated']
massInfall = f['Outputs/Output1/nodeData/massHaloEnclosedCurrent']
massBound = f['Outputs/Output1/nodeData/satelliteBoundMass']
concentration = f['Outputs/Output1/nodeData/concentration']

redshiftLastIsolated = f['Outputs/Output1/nodeData/redshiftLastIsolated']
positionOrbitalX = f['Outputs/Output1/nodeData/positionOrbitalX']
positionOrbitalY = f['Outputs/Output1/nodeData/positionOrbitalY']
positionOrbitalZ = f['Outputs/Output1/nodeData/positionOrbitalZ']
satelliteTidalHeating = f['Outputs/Output1/nodeData/satelliteTidalHeatingNormalized']
radiusVirial = f['Outputs/Output1/nodeData/darkMatterOnlyRadiusVirial']
velocityVirial = f['Outputs/Output1/nodeData/darkMatterOnlyVelocityVirial']
subhalos = (isCentral[:] == 0) & (massInfall[:] > 2.0*massResolution)
centrals = (isCentral[:] == 1)
countSubhalos = np.zeros(countTree)
for i in range(countTree):
    selectTree = (isCentral[:] == 0) & (treeIndex[:] == i+1)
    countSubhalos[i] = np.sum(weight[selectTree])
countSubhalosMean = np.mean(countSubhalos)
massHost = massInfall[centrals][0]
radiusVirialHost = radiusVirial[centrals][0]
velocityVirialHost = velocityVirial[centrals][0]
massInfallNormalized = np.log10(massInfall[subhalos]/massHost)
massBoundNormalized = np.log10(massBound[subhalos]/massInfall[subhalos])
concentrationNormalized = concentration[subhalos]
redshiftLastIsolatedNormalized = redshiftLastIsolated[subhalos]
radiusOrbitalNormalized = np.log10(np.sqrt(+positionOrbitalX[subhalos]**2+positionOrbitalY[subhalos]**2+positionOrbitalZ[subhalos]**2)/radiusVirialHost)
satelliteTidalHeatingNormalized = np.log10(1.0e-6+satelliteTidalHeating[subhalos]/velocityVirial[subhalos]**2*radiusVirial[subhalos]**2)

data=np.array(
    list(
        zip(
            massInfallNormalized,
            concentrationNormalized,
            massBoundNormalized,
            redshiftLastIsolatedNormalized,
            radiusOrbitalNormalized,
            satelliteTidalHeatingNormalized
        )
    )
)

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
augmented_normalized_data = np.hstack((normalized_data, np.expand_dims(weight[subhalos],1)))
np.random.shuffle(augmented_normalized_data)
print("Number of subhalos: "+str(len(data)))
print("Mean number of subhalos per tree: "+str(countSubhalosMean))

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

model = RealNVP(num_coupling_layers=12)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

history = model.fit(
    augmented_normalized_data, batch_size=256, epochs=200, verbose=2, validation_split=0.2
)

model.save_weights('../data/emulatorModel')

emulator = RealNVP(num_coupling_layers=12)
emulator.load_weights('../data/emulatorModel')

# From data to latent space.
z, _ = emulator(normalized_data)

# From latent space to data.
samples = emulator.distribution.sample(30000)
x, _ = emulator.predict(samples)
xt = norm_transform_inv(x, np.nanmin(x, axis = 0), np.nanmax(x, axis = 0), -1, 1)
clip = (xt[:,0] > np.log10(2.0*massResolution/massTree)) & (xt[:,2] <= 0.0) & (xt[:,2] > -xt[:,0]+np.log10(massResolution/massTree)) & (xt[:,3] >= 0.0)

# Generate a weighted subsample of the original data.
w = weight[subhalos]
i = np.arange(0, w.size, 1, dtype=int)
subsample = np.random.choice(i, size=len(xt[clip]), replace=True, p=w/np.sum(w))

# Saving emulator data to a .txt file which includes the normalization radius and non-normalized masses
np.savetxt('clipped_galacticus_data.txt', data[subsample])
np.savetxt('clipped_emulator_data.txt', xt[clip])
np.savetxt('galacticus_data.txt', data)
np.savetxt('emulator_data.txt', xt)

# create a mask that eliminates the tidal heating outlier data points from Galacticus
tidal_clip = (data[subsample, 5] > -6)

# Compute and compare the ratio of low-to-high-mass subhalos in the original data and in the emulated data.
# For the original data we weight by the subsampling weight. If this was included correctly in the training
# then the emulated data should have effectively learned these weights and produce a ratio similar to that in
# the original data.
s6 = data[:,0] > -6.0
s4 = data[:,0] > -4.0
ratioOriginal = np.sum(data[s6,0]*w[s6])/np.sum(data[s4,0]*w[s4])
print("Ratio of low-to-high-mass subhalos in original data (weighted): "+str(ratioOriginal))

ratioEmulator = np.sum(xt[clip,0] > -6.0)/np.sum(xt[clip,0] > -4.0)
print("Ratio of low-to-high-mass subhalos in emulated data: "+str(ratioEmulator))

plt.figure(figsize=(15, 10))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.legend(["train", "validation"], loc="upper right")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.ylim(-1600,200)
plt.savefig('plots/loss.png')

#f, axes = plt.subplots(5, 2)
#f.set_size_inches(15, 18)

#axes[0, 0].scatter(data[subsample, 0][tidal_clip], data[subsample, 1][tidal_clip], color="r", s=9)
#axes[0, 0].set(title="Galacticus", xlabel="Mass infall", ylabel="concentration")
#axes[0, 0].set_xlim([-6, 0])
#axes[0, 0].set_ylim([0, 23])
#axes[0, 1].scatter(xt[clip, 0][tidal_clip], xt[clip, 1][tidal_clip], color="g", s=9)
#axes[0, 1].set(title="Generated", xlabel="Mass infall", ylabel="concentration")
#axes[0, 1].set_xlim([-6, 0])
#axes[0, 1].set_ylim([0, 23])
#axes[1, 0].scatter(data[subsample, 0][tidal_clip], data[subsample, 2][tidal_clip], color="r", s=9)
#axes[1, 0].set(title="Galacticus", xlabel="Mass infall", ylabel="Mass bound")
#axes[1, 0].set_xlim([-6, 0])
#axes[1, 0].set_ylim([-5.0, 0.2])
#axes[1, 1].scatter(xt[clip, 0][tidal_clip], xt[clip, 2][tidal_clip], color="g", s=9)
#axes[1, 1].set(title="Generated", xlabel="Mass infall", ylabel="Mass bound")
#axes[1, 1].set_xlim([-6, 0])
#axes[1, 1].set_ylim([-5.0, 0.2])
#axes[2, 0].scatter(data[subsample, 0][tidal_clip], data[subsample, 3][tidal_clip], color="r", s=9)
#axes[2, 0].set(title="Galacticus", xlabel="Mass infall", ylabel="Redshift infall")
#axes[2, 0].set_xlim([-6, 0])
#axes[2, 0].set_ylim([-0.2, 6.0])
#axes[2, 1].scatter(xt[clip, 0][tidal_clip], xt[clip, 3][tidal_clip], color="g", s=9)
#axes[2, 1].set(title="Generated", xlabel="Mass infall", ylabel="Redshift infall")
#axes[2, 1].set_xlim([-6, 0])
#axes[2, 1].set_ylim([-0.2, 6.0])
#axes[3, 0].scatter(data[subsample, 0][tidal_clip], data[subsample, 4][tidal_clip], color="r", s=9)
#axes[3, 0].set(title="Galacticus", xlabel="Mass infall", ylabel="Orbital radius")
#axes[3, 0].set_xlim([-6, 0])
#axes[3, 0].set_ylim([-2.0, 1.0])
#axes[3, 1].scatter(xt[clip, 0][tidal_clip], xt[clip, 4][tidal_clip], color="g", s=9)
#axes[3, 1].set(title="Generated", xlabel="Mass infall", ylabel="Orbital radius")
#axes[3, 1].set_xlim([-6, 0])
#axes[3, 1].set_ylim([-2.0, 1.0])
#axes[4, 0].scatter(data[subsample, 0][tidal_clip], data[subsample, 5][tidal_clip], color="r", s=9)
#axes[4, 0].set(title="Galacticus", xlabel="Mass infall", ylabel="Tidal heating")
#axes[4, 0].set_xlim([-6, 0])
#axes[4, 0].set_ylim([-3.0, 5.0])
#axes[4, 1].scatter(xt[clip, 0][tidal_clip], xt[clip, 5][tidal_clip], color="g", s=9)
#axes[4, 1].set(title="Generated", xlabel="Mass infall", ylabel="Tidal heating")
#axes[4, 1].set_xlim([-6, 0])
#axes[4, 1].set_ylim([-3.0, 5.0])

def generateRealization(countSubhalosMean,emulator):
    countSubhalosRealization = np.random.poisson(countSubhalosMean,1)[0]
    samples = emulator.distribution.sample(2*countSubhalosRealization)
    x, _ = emulator.predict(samples)
    xt = norm_transform_inv(x, np.nanmin(x, axis = 0), np.nanmax(x, axis = 0), -1, 1)
    clip = np.nonzero((xt[:,0] > np.log10(2.0*massResolution/massTree)) & (xt[:,2] <= 0.0) & (xt[:,2] > -xt[:,0]+np.log10(massResolution/massTree)) & (xt[:,3] >= 0.0))
#     clip = np.nonzero((xt[:,0] > np.log10(2.0*massResolution/massTree)) & (xt[:,2] <= 0.0) & (xt[:,3] >= 0.0))
    clipLimited = clip[0][0:countSubhalosRealization-1]
    return xt[clipLimited,:]

realization=generateRealization(countSubhalosMean,emulator)
print("Number of subhalos in realization = "+str(realization.shape[0]))

# Testing now to create Density plots
from scipy.stats import gaussian_kde

concentration_density_galacticus = np.vstack([data[subsample, 0][tidal_clip], data[subsample, 1][tidal_clip]])
z1_galacticus = gaussian_kde(concentration_density_galacticus)(concentration_density_galacticus)
concentration_density_generated = np.vstack([xt[clip, 0][tidal_clip], xt[clip, 1][tidal_clip]])
z1_generated = gaussian_kde(concentration_density_generated)(concentration_density_generated)

mass_bound_density_galacticus = np.vstack([data[subsample, 0][tidal_clip], data[subsample, 2][tidal_clip]])
z2_galacticus = gaussian_kde(mass_bound_density_galacticus)(mass_bound_density_galacticus)
mass_bound_density_generated = np.vstack([xt[clip, 0][tidal_clip], xt[clip, 2][tidal_clip]])
z2_generated = gaussian_kde(mass_bound_density_generated)(mass_bound_density_generated)

mass_bound_density_galacticus = np.vstack([data[subsample, 0][tidal_clip], data[subsample, 2][tidal_clip]])
z2_galacticus = gaussian_kde(mass_bound_density_galacticus)(mass_bound_density_galacticus)
mass_bound_density_generated = np.vstack([xt[clip, 0][tidal_clip], xt[clip, 2][tidal_clip]])
z2_generated = gaussian_kde(mass_bound_density_generated)(mass_bound_density_generated)

redshift_infall_density_galacticus = np.vstack([data[subsample, 0][tidal_clip], data[subsample, 3][tidal_clip]])
z3_galacticus = gaussian_kde(redshift_infall_density_galacticus)(redshift_infall_density_galacticus)
redshift_infall_density_generated = np.vstack([xt[clip, 0][tidal_clip], xt[clip, 3][tidal_clip]])
z3_generated = gaussian_kde(redshift_infall_density_generated)(redshift_infall_density_generated)

orbital_radius_density_galacticus = np.vstack([data[subsample, 0][tidal_clip], data[subsample, 4][tidal_clip]])
z4_galacticus = gaussian_kde(orbital_radius_density_galacticus)(orbital_radius_density_galacticus)
orbital_radius_density_generated = np.vstack([xt[clip, 0][tidal_clip], xt[clip, 4][tidal_clip]])
z4_generated = gaussian_kde(orbital_radius_density_generated)(orbital_radius_density_generated)

tidal_heating_density_galacticus = np.vstack([data[subsample, 0][tidal_clip], data[subsample, 5][tidal_clip]])
z5_galacticus = gaussian_kde(tidal_heating_density_galacticus)(tidal_heating_density_galacticus)
tidal_heating_density_generated = np.vstack([xt[clip, 0][tidal_clip], xt[clip, 5][tidal_clip]])
z5_generated = gaussian_kde(tidal_heating_density_generated)(tidal_heating_density_generated)

f, axes = plt.subplots(5, 2)
f.set_size_inches(15, 18)

axes[0, 0].scatter(data[subsample, 0][tidal_clip], data[subsample, 1][tidal_clip], c = z1_galacticus, s=9)
axes[0, 0].set(title="Galacticus", xlabel="Mass infall", ylabel="concentration")
axes[0, 0].set_xlim([-6, 0])
axes[0, 0].set_ylim([0, 23])
axes[0, 1].scatter(xt[clip, 0][tidal_clip], xt[clip, 1][tidal_clip], c = z1_generated, s=9)
axes[0, 1].set(title="Generated", xlabel="Mass infall", ylabel="concentration")
axes[0, 1].set_xlim([-6, 0])
axes[0, 1].set_ylim([0, 23])
axes[1, 0].scatter(data[subsample, 0][tidal_clip], data[subsample, 2][tidal_clip], c = z2_galacticus, s=9)
axes[1, 0].set(title="Galacticus", xlabel="Mass infall", ylabel="Mass bound")
axes[1, 0].set_xlim([-6, 0])
axes[1, 0].set_ylim([-5.0, 0.2])
axes[1, 1].scatter(xt[clip, 0][tidal_clip], xt[clip, 2][tidal_clip], c = z2_generated, s=9)
axes[1, 1].set(title="Generated", xlabel="Mass infall", ylabel="Mass bound")
axes[1, 1].set_xlim([-6, 0])
axes[1, 1].set_ylim([-5.0, 0.2])
axes[2, 0].scatter(data[subsample, 0][tidal_clip], data[subsample, 3][tidal_clip], c = z3_galacticus, s=9)
axes[2, 0].set(title="Galacticus", xlabel="Mass infall", ylabel="Redshift infall")
axes[2, 0].set_xlim([-6, 0])
axes[2, 0].set_ylim([-0.2, 6.0])
axes[2, 1].scatter(xt[clip, 0][tidal_clip], xt[clip, 3][tidal_clip], c = z3_generated, s=9)
axes[2, 1].set(title="Generated", xlabel="Mass infall", ylabel="Redshift infall")
axes[2, 1].set_xlim([-6, 0])
axes[2, 1].set_ylim([-0.2, 6.0])
axes[3, 0].scatter(data[subsample, 0][tidal_clip], data[subsample, 4][tidal_clip], c = z4_galacticus, s=9)
axes[3, 0].set(title="Galacticus", xlabel="Mass infall", ylabel="Orbital radius")
axes[3, 0].set_xlim([-6, 0])
axes[3, 0].set_ylim([-2.0, 1.0])
axes[3, 1].scatter(xt[clip, 0][tidal_clip], xt[clip, 4][tidal_clip], c = z4_generated, s=9)
axes[3, 1].set(title="Generated", xlabel="Mass infall", ylabel="Orbital radius")
axes[3, 1].set_xlim([-6, 0])
axes[3, 1].set_ylim([-2.0, 1.0])
axes[4, 0].scatter(data[subsample, 0][tidal_clip], data[subsample, 5][tidal_clip], c = z5_galacticus, s=9)
axes[4, 0].set(title="Galacticus", xlabel="Mass infall", ylabel="Tidal heating")
axes[4, 0].set_xlim([-6, 0])
axes[4, 0].set_ylim([-3.0, 5.0])
axes[4, 1].scatter(xt[clip, 0][tidal_clip], xt[clip, 5][tidal_clip], c = z5_generated, s=9)
axes[4, 1].set(title="Generated", xlabel="Mass infall", ylabel="Tidal heating")
axes[4, 1].set_xlim([-6, 0])
axes[4, 1].set_ylim([-3.0, 5.0])
plt.savefig('plots/density.png')

f, axes = plt.subplots(5)
f.set_size_inches(15, 20)

axes[0].hist(data[subsample, 1][tidal_clip], bins = 70, range = (0,30), label = 'Galacticus', fill = False, edgecolor = 'blue')
axes[0].hist(xt[clip, 1][tidal_clip], bins = 70, range = (0,30), label = 'Generated', fill = False, edgecolor = 'orange')
axes[0].set(xlim = 0, ylim = 30, title = 'Concentration')
axes[0].legend()
axes[1].hist(data[subsample, 2][tidal_clip], bins = 70, range = (-2.5, 0), label = 'Galacticus', fill = False, edgecolor = 'blue')
# axes[1].hist(data[subsample, 2][tidal_clip], bins = 33, label = 'Galacticus', fill = False, edgecolor = 'blue')
axes[1].hist(xt[clip, 2][tidal_clip], bins = 70, range = (-2.5, 0), label = 'Generated', fill = False, edgecolor = 'orange')
axes[1].set(title = 'Mass Bound')
axes[1].legend()
axes[2].hist(data[subsample, 3][tidal_clip], bins = 70, range = (0, 8), label = 'Galacticus', fill = False, edgecolor = 'blue')
axes[2].hist(xt[clip, 3][tidal_clip], bins = 70, range = (0, 8), label = 'Generated', fill = False, edgecolor = 'orange')
axes[2].set(title = 'Redshift Infall')
axes[2].legend()
axes[3].hist(data[subsample, 4][tidal_clip], bins = 70, range = (-1.5, 1.5), label = 'Galacticus', fill = False, edgecolor = 'blue')
axes[3].hist(xt[clip, 4][tidal_clip], bins = 70, range = (-1.5, 1.5), label = 'Generated', fill = False, edgecolor = 'orange')
axes[3].set(title = 'Orbital Radius')
axes[3].legend()
axes[4].hist(data[subsample, 5][tidal_clip], bins = 70, range = (-4, 4), label = 'Galacticus', fill = False, edgecolor = 'blue')
axes[4].hist(xt[clip, 5][tidal_clip], bins = 70, range = (-4, 4), label = 'Generated', fill = False, edgecolor = 'orange')
axes[4].set(title = 'Tidal Heating')
axes[4].legend()
plt.savefig('plots/histograms.png')

#print(stats.ks_2samp(data[subsample, 1][tidal_clip], xt[clip, 1][tidal_clip]))
#print(stats.ks_2samp(data[subsample, 2][tidal_clip], xt[clip, 2][tidal_clip]))
#print(stats.ks_2samp(data[subsample, 3][tidal_clip], xt[clip, 3][tidal_clip]))
#print(stats.ks_2samp(data[subsample, 4][tidal_clip], xt[clip, 4][tidal_clip]))
#print(stats.ks_2samp(data[subsample, 5][tidal_clip], xt[clip, 5][tidal_clip]))

# Defining a weighted 2 sample KS test
def ks_w2(data1, data2, wei1, wei2):
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1)/sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2)/sum(wei2)])
    cdf1we = cwei1[[np.searchsorted(data1, data, side='right')]]
    cdf2we = cwei2[[np.searchsorted(data2, data, side='right')]]
    return np.max(np.abs(cdf1we - cdf2we))

print('concentration p-value: ' + str(ks_w2(data[:,1], xt[clip, 1][tidal_clip], w,np.ones(len(w[subsample][tidal_clip])))))
print('mass bound p-value: ' + str(ks_w2(data[:,2], xt[clip, 2][tidal_clip], w,np.ones(len(w[subsample][tidal_clip])))))
print('redshift infall p-value: ' + str(ks_w2(data[:,3], xt[clip, 3][tidal_clip], w, np.ones(len(w[subsample][tidal_clip])))))
print('orbital radius p-value: ' + str(ks_w2(data[:,4], xt[clip, 4][tidal_clip], w, np.ones(len(w[subsample][tidal_clip])))))
print('tidal heating p-value: ' + str(ks_w2(data[:,5], xt[clip, 5][tidal_clip], w, np.ones(len(w[subsample][tidal_clip])))))

# Creating an example Negative Binomial distribution plot
N = countSubhalosMean
s_I = 0.18
p = 1/(1 + N*s_I**2)
r = 1/s_I**2
x = np.arange(stats.nbinom.ppf(0.01, r, p),stats.nbinom.ppf(0.99, r, p))
plt.plot(x, stats.nbinom.pmf(x, r, p), 'ko', ms=1, label='nbinom pmf')
plt.savefig('plots/negative_binomial.png')
