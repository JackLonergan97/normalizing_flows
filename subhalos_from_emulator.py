from pyHalo.Halos.HaloModels.TNFW import TNFWSubhalo
from pyHalo.Halos.lens_cosmo import LensCosmo
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from pyHalo.Halos.halo_base import Halo
from pyHalo.single_realization import Realization
from pyHalo.defaults import set_default_kwargs
import random

def tnfw_mass_fraction(tau, c):
    """
    This function returns the fraction = final_mass/initial_mass, assuming a truncated NFW profile
    :param tau: the truncation radius in units of the scale radius 
    :param c: the halo concentration
    """
    x = c
    Rs = 1.0
    r_trunc = tau * Rs
    func = (r_trunc ** 2 * (-2 * x * (1 + r_trunc ** 2) + 4 * (1 + x) * r_trunc * np.arctan(x / r_trunc) -
                            2 * (1 + x) * (-1 + r_trunc ** 2) * np.log(Rs) + 2 * (1 + x) * (-1 + r_trunc ** 2) * np.log(Rs * (1 + x)) +
                            2 * (1 + x) * (-1 + r_trunc ** 2) * np.log(Rs * r_trunc) -
                            (1 + x) * (-1 + r_trunc ** 2) * np.log(Rs ** 2 * (x ** 2 + r_trunc ** 2)))) / (2. * (1 + x) * (1 + r_trunc ** 2) ** 2)
    mass_loss = func / (np.log(1+c)-c/(1+c))
    return mass_loss
    
def tau_mf_interpolator():

    N = 100
    tau = np.logspace(-2, 3, N)
    concentration = np.linspace(1.1, 100, N)
    # mass_fraction_1d = np.logspace(-1.45, -0.02, N)
    mass_fraction_1d = np.logspace(-2, -0.01, N)
    log10tau_2d = np.zeros((N, N))

    # This computes the value of tau that correponds to each pair of (concentration, mass_loss) 
    for i, con_i in enumerate(concentration):
        log10final_mass = np.log10(tnfw_mass_fraction(tau, con_i))
        mfinterp = interp1d(log10final_mass, np.log10(tau))
        for j, mass_j in enumerate(mass_fraction_1d):
            log10t = mfinterp(np.log10(mass_j))
            log10tau_2d[i,j] = log10t

    interp_points = (np.log10(concentration), np.log10(mass_fraction_1d))
    interpolator = RegularGridInterpolator(interp_points,log10tau_2d, bounds_error=False)
    return interpolator

# we will call this in the new subhalo class
truncation_radius_interpolator = tau_mf_interpolator()

class TNFWSubhaloEmulator(Halo):
    """
    Defines a truncated NFW halo that is a subhalo of the host dark matter halo
    """
    def __init__(self, infall_mass, x, y, final_bound_mass, infall_concentration, redshift, 
                 lens_cosmo_instance, unique_tag=None):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        self._lens_cosmo = lens_cosmo_instance
        
        r3d = None
        profile_definition = 'TNFW'
        sub_flag = True
        args = None
        if unique_tag is None:
            unique_tag = np.random.rand()
        
        # set the concentration
        self.c = infall_concentration
        self._bound_mass_fraction = final_bound_mass/infall_mass
        x_arcsec = x / self._lens_cosmo.cosmo.kpc_proper_per_asec(redshift)
        y_arcsec = y / self._lens_cosmo.cosmo.kpc_proper_per_asec(redshift)
        super(TNFWSubhaloEmulator, self).__init__(infall_mass, x_arcsec, y_arcsec, r3d, profile_definition, redshift, sub_flag,
                                           lens_cosmo_instance, args, unique_tag)
        
    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_kwargs_lenstronomy'):

            [concentration, rt] = self.profile_args
            Rs_angle, theta_Rs = self._lens_cosmo.nfw_physical2angle(self.mass, concentration, self.z)

            x, y = np.round(self.x, 4), np.round(self.y, 4)

            Rs_angle = np.round(Rs_angle, 10)
            theta_Rs = np.round(theta_Rs, 10)
            r_trunc_arcsec = rt / self._lens_cosmo.cosmo.kpc_proper_per_asec(self.z)

            kwargs = [{'alpha_Rs': self._rescale_norm * theta_Rs, 'Rs': Rs_angle,
                      'center_x': x, 'center_y': y, 'r_trunc': r_trunc_arcsec}]

            self._kwargs_lenstronomy = kwargs

        return self._kwargs_lenstronomy, None
        
    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):
            point = (np.log10(self.c), np.log10(self._bound_mass_fraction))
            truncation_radius_kpc = truncation_radius_interpolator(point)

            self._profile_args = (self.c, float(truncation_radius_kpc))

        return self._profile_args

    @property
    def params_physical(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_params_physical'):
            [concentration, rt] = self.profile_args
            rhos, rs, r200 = self._lens_cosmo.NFW_params_physical(self.mass, concentration, self.z)
            self._params_physical = {'rhos': rhos, 'rs': rs, 'r200': r200, 'r_trunc_kpc': rt}

        return self._params_physical
    
    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        return ['TNFW']

# Reading in the host halo mass and radius
f = h5py.File('darkMatterOnlySubHalos.hdf5', 'r')
isCentral = f['Outputs/Output1/nodeData/nodeIsIsolated']
massInfall = f['Outputs/Output1/nodeData/massHaloEnclosedCurrent']
radiusVirial = f['Outputs/Output1/nodeData/darkMatterOnlyRadiusVirial']
centrals = (isCentral[:] == 1)
massHost = massInfall[centrals][0]
radiusVirialHost = radiusVirial[centrals][0]

# Reading in emulator data
data = np.loadtxt('emulator_data.txt')

# need to create an instance of a LensCosmo class
redshift = 0.5
source_redshift = 2.0
lens_cosmo = LensCosmo(redshift, source_redshift)

# Generating inputs for Daniel's code
massInfall = massHost * (10**data[:,0])
massBound = massInfall * (10**data[:,2])
r_kpc = 1000 * radiusVirialHost * (10** data[:,4])
redshifts = [0.5] * len(massInfall)
concentration = data[:,1]
halo_list = []

# Creating a set of x,y positions
x = np.zeros(len(data))
y = np.zeros(len(data))

for i in range(len(data)):
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    
    theta = np.arccos(1 - 2*r1) # [0,pi] variable
    phi = 2 * np.pi * r2 # [0,2pi] variable
    
    x[i] = r_kpc[i] * np.cos(phi) * np.sin(theta)
    y[i] = r_kpc[i] * np.sin(phi) * np.sin(theta)

# Inputting the emulator data into Daniel's code
for i in range(len(data)):
    halo = TNFWSubhaloEmulator(massInfall[i], x[i], y[i], massBound[i], concentration[i],
                              redshifts[i], lens_cosmo)
    halo_list.append(halo)

minimum_mass_emulator = 10 ** 6
maximum_mass_emulator = 10 ** 10
kwargs_setup = {'log_mlow': np.log10(minimum_mass_emulator),
               'log_mhigh': np.log10(maximum_mass_emulator)}
prof_params = set_default_kwargs(kwargs_setup, source_redshift)
realization = Realization.from_halos(halo_list, lens_cosmo, prof_params, msheet_correction=False, rendering_classes=None)

print(realization.lensing_quantities())    
