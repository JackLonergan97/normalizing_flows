import tensorflow as tf
from itertools import chain
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import matplotlib.colors as mcolors
from scipy import stats
import h5py
import sys
import random
import time
import warnings

warnings.filterwarnings('ignore')

def norm_transform_inv(norm_data, min_val, max_val, data_min, data_max):

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

def emulator_data(emulator, lines, num_iterations):
    # removing the "\n" at the end of every line from the .txt file
    for i in range(len(lines)):
        lines[i] = lines[i].strip('\n')
        if i < 2:
            lines[i] = lines[i][1:-2]
            lines[i].split()
            lines[i] = [float(i) for i in lines[i].split()]
        else:
            lines[i] = float(lines[i])

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
    reg_x = []
    reg_y = []

    samples = [] # Array which stores each realization of a subhalo population
    num_subhalos = [0] # Array which stores the number of subhalos for a given realization. We start with 0 so when we renest the array goes from 0 to the  number of subhalos in the first realization

    ts = time.time()
    time_format = time.strftime("%H:%M:%S", time.gmtime(ts))
    sample_amount = 1.5

    # obtaining num_iteration different subhalo populations
    print('starting sampling process. Current time: ' + str(time_format))
    while n < num_iterations:
        print('starting iteration ' + str(n + 1))
        x = np.arange(stats.nbinom.ppf(0, r, p),stats.nbinom.ppf(0.9999999999999999, r, p))
        N = np.random.choice(x, p = stats.nbinom.pmf(x,r,p))
        print('starting sampling emulator points')
        sample = emulator.distribution.sample(sample_amount*N)

        # samples.append(sample) # toggling this on/off to play around with stuff
        x, _ = emulator.predict(sample)
        num_subhalos.append(len(sample))
        print('ending sampling emulator points')
 
        n += 1
#    for i in range(len(data)):
#        r1 = random.uniform(0, 1)
#        r2 = random.uniform(0, 1)
#
#        theta = np.arccos(1 - 2*r1) # [0,pi] variable
#        phi = 2 * np.pi * r2 # [0,2pi] variable
#
#        x[i] = r_kpc[i] * np.cos(phi) * np.sin(theta)
#        y[i] = r_kpc[i] * np.sin(phi) * np.sin(theta) 
    ts = time.time()
    time_format = time.strftime("%H:%M:%S", time.gmtime(ts))
    print('ending sampling process. Current time: ' + str(time_format))

    print('num_subhalos: ' + str(num_subhalos))
    print('length of num_subhalos: ' + str(len(num_subhalos)))

    # Currently, "samples" is a nested array. This part allows us to unnest the array. i.e. [[1, 2], [3, 4]] --> [1, 2, 3, 4]
    from itertools import chain

    ts = time.time()
    time_format = time.strftime("%H:%M:%S", time.gmtime(ts))
    print('beginning to predict samples. Current time: ' + str(time_format))
    # samples = list(chain(*samples))

    # We can now perform emulator.predict() since samples is a non-nested array
    # x, _ = emulator.predict(samples)

    ts = time.time()
    time_format = time.strftime("%H:%M:%S", time.gmtime(ts))
    print('finished with predicting samples. Current time: ' + str(time_format))

    # We now want to renest everything so we can apply the clip over a single realization of a subhalo population
    x_nest = [] 

    # Applying the norm_transform_inv function over each subhalo population generated from the first for loop.
    for i in range(num_iterations):
        start_index = num_subhalos[i]
        end_index = num_subhalos[i + 1]
        x_nest.append(x[start_index:end_index])
        xt = norm_transform_inv(x_nest[i], -1, 1, data_min, data_max)
        clip = (xt[:,0] > np.log10(2.0*massResolution/massTree)) & (xt[:,2] <= 0.0) & (xt[:,2] > -xt[:,0]+np.log10(massResolution/massTree)) & (xt[:,3] >= 0.0)     & (xt[:,1] > 0.0)

        if len(xt[clip]) > N:
            data = xt[clip][:int(N)]
        elif len(xt[clip]) < N:
            print('Not enough data points are being sampled in iteration ' + str(n))
            sample_amount += 0.5
            continue
        else:
            data = xt[clip]

        ts = time.time()
        time_format = time.strftime("%H:%M:%S", time.gmtime(ts))
        print('starting the append process. Current time: ' + str(time_format))
        reg_massInfall.append(massHost * (10**data[:,0]))
        reg_massBound.append(massHost * (10**data[:,2]))
        reg_concentration.append(data[:,1])
        r_kpc = 1000 * radiusVirialHost * (10**data[:,4])
        x = [0]*len(r_kpc)
        y = [0]*len(r_kpc)

        for j in range(len(data)):
            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)

            theta = np.arccos(1 - 2*r1) # [0,pi] variable
            phi = 2 * np.pi * r2 # [0,2pi] variable

            x[j] = r_kpc[j] * np.cos(phi) * np.sin(theta)
            y[j] = r_kpc[j] * np.sin(phi) * np.sin(theta)
        reg_x.append(x)
        reg_y.append(y)

        ts = time.time()
        time_format = time.strftime("%H:%M:%S", time.gmtime(ts))
        print('ending the append process. Current time: ' + str(time_format))

    return [reg_massInfall, reg_x, reg_y, reg_massBound, reg_concentration]

# Looping over each dark matter model specified in the .sh file
fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.set_size_inches(17, 5)

dm_models = sys.argv[1:]
colors = ['black', 'green']

aquarius = [] # creating an array to store mass arrays defined by Aquarius paper equation 4
mass = np.linspace(1e6, 1e10, 10000)

for model, color in zip(dm_models, colors):
    if (model != 'CDM') and (model != 'WDM') and  (model != 'CDM_cat'):
        raise Exception('The dark matter model in .sh file written after "normalizing_flows.py" was entered incorrectly!')

    necessary_data = open("necessary_data_" + model + ".txt", "r")
    lines = necessary_data.readlines()
    necessary_data.close()

    emulator = RealNVP(num_coupling_layers=12)
    emulator.load_weights('../data/emulatorModel' + model)

    # Reading in the Galacticus data
    f = h5py.File('darkMatterOnlySubHalos' + model + '.hdf5', 'r')
    mergerTreeBuildMassesGroup = f['Parameters/mergerTreeBuildMasses']
    massResolutionGroup = f['Parameters/mergerTreeMassResolution']
    massResolution = massResolutionGroup.attrs['massResolution']
    weight = f['Outputs/Output1/nodeData/nodeSubsamplingWeight']
    isCentral = f['Outputs/Output1/nodeData/nodeIsIsolated']
    nodeIsIsolated =  f['Outputs/Output1/nodeData/nodeIsIsolated']
    massInfall = f['Outputs/Output1/nodeData/basicMass'][:]
    # massInfall = f['Outputs/Output1/nodeData/massHaloEnclosedCurrent'][:]
    centrals = (isCentral[:] == 1)
    massHost = massInfall[centrals][0]

    try:
        countTree =  mergerTreeBuildMassesGroup.attrs['treeCount'][0]
    except KeyError:
        countTree = 0
        for i in range(len(massInfall)):
            if nodeIsIsolated[:][i] == 1:
                countTree += 1

    massBound = f['Outputs/Output1/nodeData/satelliteBoundMass'][:]
    subhalos = (isCentral[:] == 0) & (massInfall[:] > 2.0*massResolution)
    positionOrbitalX = f['Outputs/Output1/nodeData/positionOrbitalX']
    positionOrbitalY = f['Outputs/Output1/nodeData/positionOrbitalY']
    positionOrbitalZ = f['Outputs/Output1/nodeData/positionOrbitalZ']
    radius = np.sqrt(positionOrbitalX[subhalos]**2+positionOrbitalY[subhalos]**2+positionOrbitalZ[subhalos]**2)[:]

    data = emulator_data(emulator, lines, 1)
    em_massInfall = np.array(data[0])
    em_massBound = np.array(data[3])

    w = weight[subhalos]
    i = np.arange(0, w.size, 1, dtype=int)
    subsample = np.random.choice(i, size=len(em_massInfall), replace=True, p=w/np.sum(w))

    # gal_massInfall = np.array(massInfall[subsample])
    # gal_massBound = np.array(massBound[subsample])

    gal_infall = []
    gal_bound = []
    em_infall = []
    em_bound = []
    
    for m in mass:
        #n = np.sum(gal_massInfall > m)
        n = w[massInfall[subhalos] > m].sum()/countTree
        gal_infall.append(n)

        #n = np.sum(gal_massBound > m)
        n = w[massBound[subhalos] > m].sum()/countTree
        gal_bound.append(n)

        n = np.sum(em_massInfall > m)/countTree
        em_infall.append(n)

        n = np.sum(em_massBound > m)/countTree
        em_bound.append(n)

    # Setting up code to plot equation 4 from the Aquarius paper: N(M) = (a_0/((n + 1)*m_0^n)) * M^{n + 1}
    # the (1e13/1.8e12) factor is the conversion factor from the Aquarius host halo mass to our original host halo mass
    a_0 = 3.26e-5 / (massHost/1.8e12)
    m_0 = 2.52e7 * (massHost/1.8e12)
    n = -1.9
    aquarius.append(-(a_0/(m_0**n * (n + 1))) * mass**(n + 1))
    

    # rewriting in scientific notation for plot labels
    sci_massHost = "{:.2e}".format(massHost)
    sci_massHost = str(sci_massHost)

    ax1.plot(mass,gal_infall, '-', c = color, label = sci_massHost + '$M_{\odot}$')
    ax1.plot(mass,gal_bound, '-.', c = color, label = sci_massHost + '$M_{\odot}$')
    ax1.set_xlim(1e6, 1e10)
    ax1.set_ylim(10,5.5e5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Mass $ M $')
    ax1.set_ylabel('Number of Subhalos $ (> M) $')
    ax1.set_title('Galacticus SMF')

    ax2.plot(mass,em_infall, '-', c = color, label = sci_massHost + '$M_{\odot}$')
    ax2.plot(mass,em_bound, '-.', c = color, label = sci_massHost + '$M_{\odot}$')
    ax2.set_xlim(1e6, 1e10)
    ax2.set_ylim(10,5.5e5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Mass $ M $')
    ax2.set_ylabel('Number of Subhalos $ (> M) $')
    ax2.set_title('Emulator SMF')

print('length of mass array: ' + str(len(mass)))
print('length of aquarius array: ' + str(len(aquarius[0]))) 
ax1.plot(mass, aquarius[0], '-', c = 'cyan', label = 'Aquarius 1.00e+13 $M_{\odot}$')
# ax1.plot(mass, aquarius[1], '-', c = 'magenta', label = 'Aquarius 1.98e+12 $M_\odot$')
ax1.legend()
ax2.plot(mass, aquarius[0], '-', c = 'cyan', label = 'Aquarius 1.00e+13 $M_{\odot}$')
# ax2.plot(mass, aquarius[1], '-', c = 'magenta', label = 'Aquarius 1.98e+12 $M_\odot$')
ax2.legend()

plt.savefig('plots/quick_smf.png')

ts = time.time()
time_format = time.strftime("%H:%M:%S", time.gmtime(ts))
print('code executed! Final time: ' + str(time_format))
