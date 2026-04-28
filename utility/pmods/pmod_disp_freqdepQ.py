#!/usr/bin/env python

# *********************************************************************
# PROGRAM TO ADJUST THE ELASTIC PARAMETERS OF AN EARTH MODEL
# REFERENCED TO 1-SECOND PERIOD TO A DIFFERENT REFERENCE PERIOD
# (e.g. M2 TIDAL PERIOD) | ACCOUNTS FOR PHYSICAL DISPERSION
# AND FREQUENCY DEPENDENT Q FOLLOWING BENJAMIN ET AL. (2006)
#
# DR. MATTHEW J. SWARR APRIL 2026
#
# Copyright (c) 2014-2019: HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
#
# This file is part of LoadDef.
#
#    LoadDef is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    LoadDef is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with LoadDef.  If not, see <https://www.gnu.org/licenses/>.
#
# *********************************************************************

# IMPORT PYTHON MODULES
from __future__ import print_function
import numpy as np
from math import pi
import os
import sys

# Reference Period to Which the Model Should be Adjusted
# For example : M2 = 12.42 hours, O1 = 25.82 hours, Mf = 13.66 days
rper = 12.42*(60.*60.) # M2

# Alpha Value
# Alpha describes the degree in which Q is frequency dependent
# Benjamin et al. (2006) report alpha to vary betwen 0.2 and 0.3
alpha = 0.25

# Input Planet Model
#  :: Input Planet Model Must be of the Format: 
#  :: Radius[km], vp[km/s], vs[km/s], density[g/cc], Qk, Qmu
planet_model = ("../../input/Planet_Models/PREM.txt")

# Output Filename
outfile = ("PREM_M2TidalPeriod_FreqDepQ.txt")

# BEGIN CODE

# Ensure that the Output Directories Exist
if not (os.path.isdir("../../output/Planet_Models/")):
    os.makedirs("../../output/Planet_Models/")
 
# Read Reference Planet Model
radial_dist,vp,vs,rho,Qk,Qmu = np.loadtxt(planet_model,usecols=(0,1,2,3,4,5),unpack=True)
mu = np.multiply(np.square(vs*1000),rho*1000)
ka = np.multiply(np.square(vp*1000),rho*1000) - (4./3.)*mu
# Convert to Vp,Vs
vs = np.sqrt(np.divide(mu,rho*1000.)) / 1000.
vp = np.sqrt(np.divide((ka + (4./3.)*mu),rho*1000.)) / 1000.
QmuInv = np.divide(1,Qmu)
QkInv = np.divide(1,Qk)
print(('mu[orig]', mu))
print(('ka[orig]', ka))
print(('vp[orig]', vp))
print(('vs[orig]', vs))

# Computing the perturbation to mu and kappa using Benajmin et al.'s model for frequency depdendent Q.
# wm = frequency in which below Q is frequency dependent (i.e., at tidal frequencies)
wm = (2 * pi) * 3.09e-4 
dmu = np.multiply(mu,QmuInv) * ((2/pi) * (np.log(wm/rper) + (1/alpha) * (1 - (wm/rper)**alpha)))
newmu = mu+dmu # Note that 'newmu' should be ~ equivalent to 'mu_pert'
dka = np.multiply(ka,QkInv) * ((2/pi) * (np.log(wm/rper) + (1/alpha) * (1 - (wm/rper)**alpha)))
newka = ka+dka
print(('mu[D&T]', newmu))
print(('ka[D&T]', newka))
vp_pert = np.sqrt(np.divide((newka + (4./3.)*newmu),rho*1000)) / 1000.
vs_pert = np.sqrt(np.divide((newmu),rho*1000)) / 1000.
print(('vp[D&T]', vp_pert))
print(('vs[D&T]', vs_pert))

# Write to File
fname = ("../../output/Planet_Models/" + outfile)
params = np.column_stack((radial_dist,vp_pert,vs_pert,rho,Qk,Qmu))
#f_handle = open(fname,'w')
np.savetxt(fname,params,fmt='%f %f %f %f %f %f')
#f_handle.close()

