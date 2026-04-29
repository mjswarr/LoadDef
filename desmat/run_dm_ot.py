#!/usr/bin/env python

# *********************************************************************
# MAIN PROGRAM TO COMPUTE A DESIGN MATRIX TO INVERT FOR OCEAN TIDAL LOAD --
# BY CONVOLVING DISPLACEMENT LOAD GREENS FUNCTIONS WITH A UNIFORM REAL 
# AND IMAGINARY LOAD IN EACH USER-DEFINED GRID CELL 
#
# DR. MATTHEW J. SWARR APRIL 2026
#
# Copyright (c) 2014-2023: HILARY R. MARTENS, LUIS RIVERA, MARK SIMONS         
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

# IMPORT PRINT FUNCTION
from __future__ import print_function

# IMPORT MPI MODULE
from mpi4py import MPI

# MODIFY PYTHON PATH TO INCLUDE 'LoadDef' DIRECTORY
import sys
import os
sys.path.append(os.getcwd() + "/../")

# IMPORT PYTHON MODULES
import numpy as np
import scipy as sc
import datetime
import netCDF4 
from math import pi
from scipy import interpolate
from CONVGF.utility import read_station_file
from CONVGF.utility import read_lsmask
from CONVGF.utility import read_greens_fcn_file
from CONVGF.utility import read_greens_fcn_file_norm
from CONVGF.utility import normalize_greens_fcns
from CONVGF.CN import load_convolution
from CONVGF.CN import compute_specific_greens_fcns
from CONVGF.CN import generate_integration_mesh
from CONVGF.CN import intmesh2geogcoords
from CONVGF.CN import integrate_greens_fcns
from CONVGF.CN import compute_angularDist_azimuth
from CONVGF.CN import interpolate_lsmask
from CONVGF.CN import coef2amppha
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --------------- SPECIFY USER INPUTS --------------------- #

# Reference Frame (used for filenames) [Blewitt 2003]
rfm = "cf"

# Greens Function File
#  :: May be load Green's function file output directly from run_gf.py (norm_flag = False)
#  :: May be from a published table, normalized according to Farrell (1972) conventions [theta, u_norm, v_norm]
pmod = "PREM"
grn_file = ("../output/Greens_Functions/" + rfm + "_" + pmod + ".txt")
norm_flag  = False

# Full Path to Grid File Containing Cells
#  :: Format: south lat [float], north lat [float], west lon [float], east lon [float], unique cell id [string]
cellfname = ("cells_28.0_50.0_233.0_258.0_0.25")
loadgrid  = ("../output/Grid_Files/nc/cells/" + cellfname + ".nc") 

# NEW OPTION: Provide a common geographic mesh? 
# If True, must provide the full path to a mesh file (see: GRDGEN/common_mesh). 
# If False, a station-centered grid will be created within the functions called here. 
common_mesh = True
# Full Path to Grid File Containing Surface Mesh (for sampling the load Green's functions)
#  :: Format: latitude midpoints [float,degrees N], longitude midpoints [float,degrees E], unit area of each patch [float,dimensionless (need to multiply by r^2)]
meshfname = ("commonMesh_regional_28.0_50.0_233.0_258.0_0.01_0.01_oceanmask")
convmesh = ("../output/Grid_Files/nc/commonMesh/" + meshfname + ".nc")

# Planet Radius (in meters; used for Greens function normalization)
planet_radius = 6371000.

# Load Density
#  Recommended: 1000 kg/m^3 as a standard for water-mass inversion
ldens = 1000.0

# Ocean/Land Mask
#  :: 0 = do not mask ocean or land (retain full model); 1 = mask out land (retain ocean); 2 = mask out oceans (retain land)
#  :: Recommended: 1 for oceanic; 2 for atmospheric and continental water
#  :: When pre-generating a common mesh, a land-sea mask can be applied a priori to the mesh. If that is done, it is recommended to set lsmask_type = 0 here. 
lsmask_type = 0

# Full Path to Land-Sea Mask File (May be Irregular and Sparse)
#  :: Format: Lat, Lon, Mask [0=ocean; 1=land]
lsmask_file = ("../input/Land_Sea/ETOPO1_Ice_g_gmt4_wADD.txt")

# Station/Grid-Point Location File (Lat, Lon, StationName)
sta_file_name = ("NOTA_Select")
sta_file = ("../input/Station_Locations/" + sta_file_name + ".txt")

# Optional: Additional string to include in output filenames (e.g. "_2019")
if (common_mesh == True):
    mtag = meshfname
else:
    mtag = "stationMesh"
outstr = (pmod + "_" + cellfname + "_" + mtag + "_" + sta_file_name)

# ------------------ END USER INPUTS ----------------------- #

# -------------------- SETUP MPI --------------------------- #

# Get the Main MPI Communicator That Controls Communication Between Processors
comm = MPI.COMM_WORLD
# Get My "Rank", i.e. the Processor Number Assigned to Me
rank = comm.Get_rank()
# Get the Total Number of Other Processors Used
size = comm.Get_size()

# ---------------------------------------------------------- #

# -------------------- BEGIN CODE -------------------------- #

# LoadFile Format ("bbox" tells the software to read the text file line by line for individual bounding-box cells in the load grid)
loadfile_format = "bbox"

# Bounding boxes for grid cells are regular in lat/lon
regular = True

# Check for existence of load grid
if not os.path.isfile(loadgrid):
    sys.exit('Error: The load grid does not exist. You may need to create it. See GRDGEN/design_matrix/ .')

# Put loadgrid file into a list (for consistency with how traditional load files are treated)
load_files = []
load_files.append(loadgrid)

# Ensure that the Output Directories Exist
if (rank == 0):
    if not (os.path.isdir("../output/Convolution/")):
        os.makedirs("../output/Convolution/")
    if not (os.path.isdir("../output/DesignMatrixLoad/")):
        os.makedirs("../output/DesignMatrixLoad/")
    if not (os.path.isdir("../output/Figures/")):
        os.makedirs("../output/Figures/")

    # Read Station File
    slat,slon,sta = read_station_file.main(sta_file)

    # Ensure that Station Locations are in Range 0-360
    neglon_idx = np.where(slon<0.)
    slon[neglon_idx] += 360.

    # Determine Number of Stations Read In
    if isinstance(slat,float) == True: # only 1 station
        numel = 1
    else:
        numel = len(slat)

    # Generate an Array of File Indices
    sta_idx = np.linspace(0,numel,num=numel,endpoint=False)
    np.random.shuffle(sta_idx)

else: # If I'm a worker, I know nothing yet about the data
    slat = slon = sta = numel = sta_idx = None

# Make Sure Everyone Has Reported Back Before Moving On
comm.Barrier()

# All Processors Get Certain Arrays and Parameters; Broadcast Them
sta          = comm.bcast(sta, root=0)
slat         = comm.bcast(slat, root=0)
slon         = comm.bcast(slon, root=0)
numel        = comm.bcast(numel, root=0)
sta_idx      = comm.bcast(sta_idx, root=0)

# MPI: Determine the Chunk Sizes for the Convolution
total_stations = len(slat)
nominal_load = total_stations // size # Floor Divide
# Final Chunk Might Be Different in Size Than the Nominal Load
if rank == size - 1:
    procN = total_stations - rank * nominal_load
else:
    procN = nominal_load

# Make some preparations that are common to all stations
if (rank == 0): 

    # Read in the Land-Sea Mask
    if (lsmask_type > 0):
        lslat,lslon,lsmask = read_lsmask.main(lsmask_file)
    else:
        # Doesn't really matter so long as there are some values filled in with something other than 1 or 2
        lat1d = np.arange(-90.,90.,5.)
        lon1d = np.arange(0.,360.,5.)
        olon,olat = np.meshgrid(lon1d,lat1d)
        lslat = olat.flatten()
        lslon = olon.flatten()
        lsmask = np.ones((len(lslat),)) * -1.

    # Ensure that Land-Sea Mask Longitudes are in Range 0-360
    neglon_idx = np.where(lslon<0.)
    lslon[neglon_idx] += 360.
 
    # Read in the loadgrid
    #  Here, used for determining and writing out the center of each load cell
    lcext = loadgrid[-2::]
    if (lcext == 'xt'):
        load_cells = np.loadtxt(loadgrid,usecols=(4,),unpack=True,dtype='U')
        lcslat,lcnlat,lcwlon,lcelon = np.loadtxt(loadgrid,usecols=(0,1,2,3),unpack=True)
    elif (lcext == 'nc'):
        f = netCDF4.Dataset(loadgrid)
        load_cells = f.variables['cell_ids'][:]
        lcslat = f.variables['slatitude'][:]
        lcnlat = f.variables['nlatitude'][:]
        lcwlon = f.variables['wlongitude'][:]
        lcelon = f.variables['elongitude'][:]
        f.close()    
    # Ensure that Bounding Box Longitudes are in Range 0-360
    for yy in range(0,len(lcwlon)):
        if (lcwlon[yy] < 0.):
            lcwlon[yy] += 360.
        if (lcelon[yy] < 0.):
            lcelon[yy] += 360.
    # Compute center of each load cell
    print(':: Warning: Computing center of load cells. Special consideration should be made for cells spanning the prime meridian, if applicable.')
    # For prime meridian, should shift longitude range to [-180,180]. Still need to do this...
    lclat = (lcslat + lcnlat)/2.
    lclon = (lcwlon + lcelon)/2.

    # Read in the common convolution mesh, if applicable
    if (common_mesh == True): 
        print(':: Common Mesh True. Reading in ilat, ilon, iarea.')
        lcext = convmesh[-2::]
        if (lcext == 'xt'):
            ilat,ilon,unit_area = np.loadtxt(convmesh,usecols=(0,1,2),unpack=True)
            iarea = np.multiply(unit_area, planet_radius**2) # convert from unit area to true area of the spherical patch in m^2
        elif (lcext == 'nc'):
            f = netCDF4.Dataset(convmesh)
            ilat = f.variables['midpoint_lat'][:]
            ilon = f.variables['midpoint_lon'][:]
            unit_area = f.variables['unit_area_patch'][:]
            f.close()
            iarea = np.multiply(unit_area, planet_radius**2) # convert from unit area to true area of the spherical patch in m^2
    else:
        ilat = ilon = iarea = None

    # Determine the Land-Sea Mask: Interpolate onto Mesh
    if (common_mesh == True): 
        print(':: Common Mesh True. Applying Land-Sea Mask.')
        print(':: Number of Grid Points: %s | Size of LSMask: %s' %(str(len(ilat)), str(lsmask.shape)))
        lsmk = interpolate_lsmask.main(ilat,ilon,lslat,lslon,lsmask)
        print(':: Finished LSMask Interpolation.')
    else:
        lsmk = None

    # For a common mesh, can pre-determine the mesh points within each cell used for the inversion.
    # Also, apply the land-sea mask.
    if (common_mesh == True):
        ## Check load file format
        if not (loadfile_format == "bbox"): # list of cells, rather than traditional load files
            sys.exit(':: Error -- The loadfile format should be bbox for generating the design matrix for surface load distribution.')
        ## Prepare land-sea mask application
        if (lsmask_type == 2): 
            test_elements = np.where(lsmk == 0); test_elements = test_elements[0]
        elif (lsmask_type == 1): 
            test_elements = np.where(lsmk == 1); test_elements = test_elements[0]
        ## Select the Appropriate Cell ID
        count = 0 # initialize figure counter (don't want to get stuck in huge loop plotting figures...)
        colvals = np.empty((len(lclat),),dtype=object) # initialize array that will hold lists of column indices (corresponding to LGF mesh points in each grid cell)
        for qq in range(0,len(lclat)):
            clc = load_cells[qq]
            cslat = lcslat[qq]
            cnlat = lcnlat[qq]
            cwlon = lcwlon[qq]
            celon = lcelon[qq]
            ## Find ilat and ilon within cell
            yes_idx = np.where((ilat >= cslat) & (ilat <= cnlat) & (ilon >= cwlon) & (ilon <= celon)); yes_idx = yes_idx[0]
            print(':: Number of convolution grid points within load cell ', clc, ' of ', len(lclat), ': ', len(yes_idx))
            ## Apply land-sea mask
            if (lsmask_type == 2): 
                mask = np.isin(yes_idx, test_elements, assume_unique=True)
                idx_to_delete = np.nonzero(mask)
                yes_idx = np.delete(yes_idx,idx_to_delete)
            elif (lsmask_type == 1): 
                mask = np.isin(yes_idx, test_elements, assume_unique=True)
                idx_to_delete = np.nonzero(mask)
                yes_idx = np.delete(yes_idx,idx_to_delete)
            ## Write indices within cell to the main array
            colvals[qq] = yes_idx.tolist()
            #### OPTIONAL: Plot the load cell
            #### To suppress plotting, set max_count to 0
            #max_count = 0
            #if (len(yes_idx)>0):
            #    if (count < max_count):
            #        plot_fig = True
            #        count += 1
            #        print(':: Figure count: ', count)
            #    else:
            #        plot_fig = False
            #else:
            #    plot_fig = False
            #if plot_fig:
            #    print(':: Plotting the load cell. [run_dm_load.py]')
            #    cslat_plot = cslat - 0.5
            #    cnlat_plot = cnlat + 0.5
            #    cwlon_plot = cwlon - 0.5
            #    celon_plot = celon + 0.5
            #    idx_plot = np.where((ilon >= cwlon_plot) & (ilon <= celon_plot) & (ilat >= cslat_plot) & (ilat <= cnlat_plot)); idx_plot = idx_plot[0]
            #    ilon_plot = ilon[idx_plot]
            #    ilat_plot = ilat[idx_plot]
            #    ic1_plot = cic1[idx_plot]
            #    plt.scatter(ilon_plot,ilat_plot,c=ic1_plot,s=1,cmap=cm.BuPu)
            #    plt.colorbar(orientation='horizontal')
            #    fig_name = ("../output/Figures/" + str(cslat) + "_" + str(cnlat) + "_" + str(cwlon) + "_" + str(celon) + ".png")
            #    plt.savefig(fig_name,format="png")
            #    #plt.show()
            #    plt.close()
    else:
        colvals = None

    # Initialize Arrays
    numcells = len(lclat)
    eamp_re = np.empty((numel,numcells))
    epha_re = np.empty((numel,numcells))
    namp_re = np.empty((numel,numcells))
    npha_re = np.empty((numel,numcells))
    vamp_re = np.empty((numel,numcells))
    vpha_re = np.empty((numel,numcells))

    eamp_im = np.empty((numel,numcells))
    epha_im = np.empty((numel,numcells))
    namp_im = np.empty((numel,numcells))
    npha_im = np.empty((numel,numcells))
    vamp_im = np.empty((numel,numcells))
    vpha_im = np.empty((numel,numcells))

# If I'm a Worker, I Know Nothing About the Data
else:
    load_cells = lclat = lclon = lslat = lslon = lsmask = None 
    ilat = ilon = iarea = lsmk = colvals = numcells = None
    eamp_re = epha_re = namp_re = npha_re = vamp_re = vpha_re = None
    eamp_im = epha_im = namp_im = npha_im = vamp_im = vpha_im = None

# Make Sure Everyone Has Reported Back Before Moving On
comm.Barrier()

# All Processors Get Certain Arrays and Parameters; Broadcast Them
load_cells  = comm.bcast(load_cells, root=0)
lclat       = comm.bcast(lclat, root=0)
lclon       = comm.bcast(lclon, root=0)
lslat       = comm.bcast(lslat, root=0)
lslon       = comm.bcast(lslon, root=0)
lsmask      = comm.bcast(lsmask, root=0)
ilat        = comm.bcast(ilat, root=0)
ilon        = comm.bcast(ilon, root=0)
iarea       = comm.bcast(iarea, root=0)
lsmk        = comm.bcast(lsmk, root=0)
colvals     = comm.bcast(colvals, root=0)
numcells    = comm.bcast(numcells, root=0)
eamp_re         = comm.bcast(eamp_re, root=0)
epha_re         = comm.bcast(epha_re, root=0)
namp_re         = comm.bcast(namp_re, root=0)
npha_re         = comm.bcast(npha_re, root=0)
vamp_re         = comm.bcast(vamp_re, root=0)
vpha_re         = comm.bcast(vpha_re, root=0)
eamp_im         = comm.bcast(eamp_im, root=0)
epha_im         = comm.bcast(epha_im, root=0)
namp_im         = comm.bcast(namp_im, root=0)
npha_im         = comm.bcast(npha_im, root=0)
vamp_im         = comm.bcast(vamp_im, root=0)
vpha_im         = comm.bcast(vpha_im, root=0)


# Gather the Processor Workloads for All Processors
sendcounts = comm.gather(procN, root=0)

# Create a Data Type for the Convolution Results
cntype = MPI.DOUBLE.Create_contiguous(1)
cntype.Commit()

# Create a Data Type for Convolution Results for each Station and Load File
ltype = MPI.DOUBLE.Create_contiguous(numcells)
ltype.Commit()

# Scatter the Station Locations (By Index)
d_sub = np.empty((procN,))
comm.Scatterv([sta_idx, (sendcounts, None), cntype], d_sub, root=0)

# Set up the arrays
eamp_re_sub = np.empty((len(d_sub),numcells))
epha_re_sub = np.empty((len(d_sub),numcells))
namp_re_sub = np.empty((len(d_sub),numcells))
npha_re_sub = np.empty((len(d_sub),numcells))
vamp_re_sub = np.empty((len(d_sub),numcells))
vpha_re_sub = np.empty((len(d_sub),numcells))

eamp_im_sub = np.empty((len(d_sub),numcells))
epha_im_sub = np.empty((len(d_sub),numcells))
namp_im_sub = np.empty((len(d_sub),numcells))
npha_im_sub = np.empty((len(d_sub),numcells))
vamp_im_sub = np.empty((len(d_sub),numcells))
vpha_im_sub = np.empty((len(d_sub),numcells))

# Set up Design matrix (rows = stations[e_re,e_im,n_re,n_im,u_re,u_im]; columns = load cells * 2 (one for a real load and one for an imaginary load))
if (rank == 0):
    desmat = np.zeros((numel*6, numcells*2)) # Multiplication by 6 for 6 spatial dimensions (e_re,e_im,n_re,e_im,u_re,u_im)
    dmrows = np.empty((numel*6,),dtype='U10') # Assumes that station names are no more than 9 characters in length (with E_re, E_im, N_re, N_im, or U_re, U_im also appended)
    sclat = np.zeros((numel*6,))
    sclon = np.zeros((numel*6,))

# Loop Through Each Station
for ii in range(0,len(d_sub)):
 
    # Current station
    current_sta = int(d_sub[ii]) # Index

    # Remove Index If Only 1 Station
    if (numel == 1): # only 1 station read in
        csta = sta
        clat = slat
        clon = slon
    else:
        csta = sta[current_sta]
        clat = slat[current_sta]
        clon = slon[current_sta]

    # If Rank is Main, Output Station Name
    try:
        csta = csta.decode()
    except:
        pass

    # Output File Name
    cnv_out = csta + "_" + rfm + "_" + outstr + ".txt"

    # Status update
    print(':: Working on station: %s | Number: %6d of %6d | Rank: %6d' %(csta, (ii+1), len(d_sub), rank))

    # If using a common mesh, then integrate the Green's Functions and sum up all the cells
    if (common_mesh == True):

        print(':: Common Mesh True. Computing specific Greens functions for common mesh.')
        # Read in the Green's Functions
        if norm_flag == True:
            theta,u,v,unormFarrell,vnormFarrell = read_greens_fcn_file_norm.main(grn_file,rad)
        else:
            theta,u,v,unormFarrell,vnormFarrell = read_greens_fcn_file.main(grn_file)
        # Normalize Green's According to Farrell Convention
        nfactor = 1E12*planet_radius
        unorm = np.multiply(u,theta) * nfactor
        vnorm = np.multiply(v,theta) * nfactor
        # Interpolate Green's Functions
        tck_gfu = interpolate.splrep(theta,unorm,k=3)
        tck_gfv = interpolate.splrep(theta,vnorm,k=3)
        # Find Great-Circle Distances between Station and Grid Points in the Common Mesh
        delta,haz = compute_angularDist_azimuth.main(clat,clon,ilat,ilon)
        # Compute Integrated Greens Functions
        gfu = interpolate.splev(delta,tck_gfu,der=0)
        gfv = interpolate.splev(delta,tck_gfv,der=0)
        uint = iarea * gfu
        vint = iarea * gfv
        # Un-normalize
        uint = np.divide(uint,delta) / nfactor
        vint = np.divide(vint,delta) / nfactor
        # Compute Greens Functions Specific to Receiver and Grid (Geographic Coordinates)
        # Per the small-angle approximation, when the width of the cell is small, then the integrals over the horizontal displacement response
        #  reduce to [-beta*cos(alpha)] for the north component and [-beta*sin(alpha)] for the east component. 
        #  See equations 4.221 and 4.222 in H.R. Martens (2016, Caltech thesis). Here, the T(alpha) function is included in the integration. 
        #  When we use a common mesh, the convolution mesh is no longer symmetric about the station; therefore, it does not make sense to 
        #  include T(alpha) in the integration. However, we can see that the T(alpha) function can be moved outside the integral when the 
        #  value of beta is small (and we can invoke the small-angle approximation): 2*sin(beta/2) reduces to 2*(beta/2) = beta.
        #  In this way, we treat the horizontal-component integration in the same way as the vertical component:
        #  Integrate over the area of each patch (area element on a sphere): int_theta int_phi r^2 sin(theta) d(theta) d(phi)
        #   where theta is co-latitude and phi is azimuth in a geographic coordinate system.
        #  Next, we can multiply the horizontal solutions by [-cos(alpha)] and [-sin(alpha)] to convert to north and east components, respectively,
        #   where alpha is the azimuthal angle between the north pole and the load point, as subtended by the station, and as measured clockwise from north. 
        ur,ue,un = compute_specific_greens_fcns.main(haz,uint,vint)

        print(':: Common Mesh True. Summing up integrated and specific LGFs within each cell directly.')
        ec2 = 0 # imaginary component is zero
        nc2 = 0 # imaginary component is zero
        vc2 = 0 # imaginary component is zero
        # For each station, must sum up the specific LGFs and load values for every inversion cell
        for dd in range(0,numcells): # loop through load cells used in the inversion grid
            mycols = colvals[dd] # Find indices of Greens-function mesh points within the current load cell
            if (len(mycols) == 0): # Nothing in this cell; assign everything to zero.
                # Assign values to appropriate amplitude arrays
                eamp_re_sub[ii,dd] = 0.
                epha_re_sub[ii,dd] = 0.
                namp_re_sub[ii,dd] = 0.
                npha_re_sub[ii,dd] = 0.
                vamp_re_sub[ii,dd] = 0.
                vpha_re_sub[ii,dd] = 0.
                eamp_im_sub[ii,dd] = 0.
                epha_im_sub[ii,dd] = 0.
                namp_im_sub[ii,dd] = 0.
                npha_im_sub[ii,dd] = 0.
                vamp_im_sub[ii,dd] = 0.
                vpha_im_sub[ii,dd] = 0.
            else: # Sum up the contributions from each surface patch within the current load cell
                ec1 = np.sum(ue[mycols])*ldens # Sum up all the relevant integrated and specific Greens functions (east); and then multiply by load density
                nc1 = np.sum(un[mycols])*ldens # Sum up all the relevant integrated and specific Greens functions (north); and then multiply by load density
                vc1 = np.sum(ur[mycols])*ldens # Sum up all the relevant integrated and specific Greens functions (up); and then multiply by load density
                # Convert Coefficients to Amplitude and Phase
                # Note: Conversion from meters to mm also happens here!
                ceamp,cepha,cnamp,cnpha,cvamp,cvpha = coef2amppha.main(ec1,ec2,nc1,nc2,vc1,vc2)
                # Assign values to appropriate amplitude arrays
                eamp_re_sub[ii,dd] = ceamp
                epha_re_sub[ii,dd] = cepha
                namp_re_sub[ii,dd] = cnamp
                npha_re_sub[ii,dd] = cnpha
                vamp_re_sub[ii,dd] = cvamp
                vpha_re_sub[ii,dd] = cvpha
                eamp_im_sub[ii,dd] = ceamp
                epha_im_sub[ii,dd] = cepha + 90.
                namp_im_sub[ii,dd] = cnamp
                npha_im_sub[ii,dd] = cnpha + 90.
                vamp_im_sub[ii,dd] = cvamp
                vpha_im_sub[ii,dd] = cvpha + 90.

    # If not using a common mesh, then set up a station-centered grid and run the convolution as normal
    else:
        # For a station-centered grid, we cannot pre-determine the grid points within each cell, so we will send information to another function. 
        #### NOTE: Mesh defaults are adjusted to ensure we get a good number of points within each grid cell to adequately represent the shape of each cell.
        print(':: Common Mesh False. Performing the standard convolution.')
        # Compute Convolution for Current File
        eamp_re_sub[ii,:],epha_re_sub[ii,:],namp_re_sub[ii,:],npha_re_sub[ii,:],vamp_re_sub[ii,:],vpha_re_sub[ii,:] = load_convolution.main(\
            grn_file,norm_flag,load_files,loadfile_format,regular,lslat,lslon,lsmask,lsmask_type,\
            clat,clon,csta,cnv_out,load_density=ldens,azminc=0.5,delinc3=0.005,delinc4=0.02,delinc5=0.05)

        eamp_im_sub[ii,:] = eamp_re_sub[ii,:]
        epha_im_sub[ii,:] = epha_re_sub[ii,:] + 90.
        namp_im_sub[ii,:] = namp_re_sub[ii,:]
        npha_im_sub[ii,:] = npha_re_sub[ii,:] + 90.
        vamp_im_sub[ii,:] = vamp_re_sub[ii,:]
        vpha_im_sub[ii,:] = vpha_re_sub[ii,:] + 90.

# Make Sure All Jobs Have Finished Before Continuing
comm.Barrier()

# Gather Results
comm.Gatherv(eamp_re_sub, [eamp_re, (sendcounts, None), ltype], root=0)
comm.Gatherv(epha_re_sub, [epha_re, (sendcounts, None), ltype], root=0)
comm.Gatherv(namp_re_sub, [namp_re, (sendcounts, None), ltype], root=0)
comm.Gatherv(npha_re_sub, [npha_re, (sendcounts, None), ltype], root=0)
comm.Gatherv(vamp_re_sub, [vamp_re, (sendcounts, None), ltype], root=0)
comm.Gatherv(vpha_re_sub, [vpha_re, (sendcounts, None), ltype], root=0)
comm.Gatherv(eamp_im_sub, [eamp_im, (sendcounts, None), ltype], root=0)
comm.Gatherv(epha_im_sub, [epha_im, (sendcounts, None), ltype], root=0)
comm.Gatherv(namp_im_sub, [namp_im, (sendcounts, None), ltype], root=0)
comm.Gatherv(npha_im_sub, [npha_im, (sendcounts, None), ltype], root=0)
comm.Gatherv(vamp_im_sub, [vamp_im, (sendcounts, None), ltype], root=0)
comm.Gatherv(vpha_im_sub, [vpha_im, (sendcounts, None), ltype], root=0)

# Make Sure Everyone Has Reported Back Before Moving On
comm.Barrier()

# Free Data Type
cntype.Free()
ltype.Free()

# Re-organize Solutions
if (rank == 0):
    narr,nidx = np.unique(sta_idx,return_index=True)
    try:
        eamp_re = eamp_re[nidx,:]; namp_re = namp_re[nidx,:]; vamp_re = vamp_re[nidx,:]
        epha_re = epha_re[nidx,:]; npha_re = npha_re[nidx,:]; vpha_re = vpha_re[nidx,:]
        eamp_im = eamp_im[nidx,:]; namp_im = namp_im[nidx,:]; vamp_im = vamp_im[nidx,:]
        epha_im = epha_im[nidx,:]; npha_im = npha_im[nidx,:]; vpha_im = vpha_im[nidx,:]
    except:
        eamp_re = eamp_re[nidx]; namp_re = namp_re[nidx]; vamp_re = vamp_re[nidx]
        epha_re = epha_re[nidx]; npha_re = npha_re[nidx]; vpha_re = vpha_re[nidx]
        eamp_im = eamp_im[nidx]; namp_im = namp_im[nidx]; vamp_im = vamp_im[nidx]
        epha_im = epha_im[nidx]; npha_im = npha_im[nidx]; vpha_im = vpha_im[nidx]
    #print('Up amplitude (rows = stations; cols = load cells):')
    #print(vamp)
    #print('Up phase (rows = stations; cols = load cells):')
    #print(vpha)

# Loop Through Each Station & Populate the Design Matrix
for jj in range(0,len(slat)):

    if (rank == 0):

        # Remove Index If Only 1 Station
        if (numel == 1): # only 1 station read in
            csta = sta
            clat = slat
            clon = slon
        else:
            csta = sta[jj]
            clat = slat[jj]
            clon = slon[jj]

        # Convert Amp/Pha to Displacement
        edisp_re = np.multiply(eamp_re[jj,:],np.cos(np.multiply(epha_re[jj,:],(np.pi/180.))))
        ndisp_re = np.multiply(namp_re[jj,:],np.cos(np.multiply(npha_re[jj,:],(np.pi/180.))))
        udisp_re = np.multiply(vamp_re[jj,:],np.cos(np.multiply(vpha_re[jj,:],(np.pi/180.))))
        edisp_im = np.multiply(eamp_im[jj,:],np.cos(np.multiply(epha_im[jj,:],(np.pi/180.))))
        ndisp_im = np.multiply(namp_im[jj,:],np.cos(np.multiply(npha_im[jj,:],(np.pi/180.))))
        udisp_im = np.multiply(vamp_im[jj,:],np.cos(np.multiply(vpha_im[jj,:],(np.pi/180.))))

        # Fill in Design Matrix
        idxe_re = (jj*6)+0
        idxe_im = (jj*6)+1
        idxn_re = (jj*6)+2
        idxn_im = (jj*6)+3
        idxu_re = (jj*6)+4
        idxu_im = (jj*6)+5
        desmat[idxe_re,:] = edisp_re
        desmat[idxn_re,:] = ndisp_re
        desmat[idxu_re,:] = udisp_re
        desmat[idxe_im,:] = edisp_im
        desmat[idxn_im,:] = ndisp_im
        desmat[idxu_im,:] = udisp_im
        dmrows[idxe_re] = (csta + 'E_re')
        dmrows[idxn_re] = (csta + 'N_re')
        dmrows[idxu_re] = (csta + 'U_re')
        dmrows[idxe_im] = (csta + 'E_im')
        dmrows[idxn_im] = (csta + 'N_im')
        dmrows[idxu_im] = (csta + 'U_im')
        sclat[idxe_re] = clat_re
        sclat[idxn_re] = clat_re
        sclat[idxu_re] = clat_re
        sclat[idxe_im] = clat_im
        sclat[idxn_im] = clat_im
        sclat[idxu_im] = clat_im
        sclon[idxe_re] = clon_re
        sclon[idxn_re] = clon_re
        sclon[idxu_re] = clon_re
        sclon[idxe_im] = clon_im
        sclon[idxn_im] = clon_im
        sclon[idxu_im] = clon_im

# Write Design Matrix to File
if (rank == 0):
    print(":: Writing netCDF-formatted file.")
    f_out = ("designmatrix_" + rfm + "_" + outstr + ".nc")
    f_file = ("../output/DesignMatrixLoad/" + f_out)
    # Open new NetCDF file in "write" mode
    dataset = netCDF4.Dataset(f_file,'w',format='NETCDF4_CLASSIC')
    # Define dimensions for variables
    desmat_shape = desmat.shape
    num_rows = desmat_shape[0]
    num_cols = desmat_shape[1]
    nstacomp = dataset.createDimension('nstacomp',num_rows)
    nloadcell = dataset.createDimension('nloadcell',num_cols)
    nchars = dataset.createDimension('nchars',10)
    # Create variables
    sta_comp_id = dataset.createVariable('sta_comp_id','S1',('nstacomp','nchars'))
    load_cell_id = dataset.createVariable('load_cell_id','S1',('nloadcell','nchars'))
    design_matrix = dataset.createVariable('design_matrix',float,('nstacomp','nloadcell'))
    sta_comp_lat = dataset.createVariable('sta_comp_lat',float,('nstacomp',))
    sta_comp_lon = dataset.createVariable('sta_comp_lon',float,('nstacomp',))
    load_cell_lat = dataset.createVariable('load_cell_lat',float,('nloadcell',))
    load_cell_lon = dataset.createVariable('load_cell_lon',float,('nloadcell',))
    # Add units
    sta_comp_id.units = 'string'
    sta_comp_id.long_name = 'station_component_id'
    load_cell_id.units = 'string'
    load_cell_id.long_name = 'load_cell_id'
    design_matrix.units = 'mm'
    design_matrix.long_name = 'displacement_mm'
    sta_comp_lat.units = 'degrees_north'
    sta_comp_lat.long_name = 'station_latitude'
    sta_comp_lon.units = 'degrees_east'
    sta_comp_lon.long_name = 'station_longitude'
    load_cell_lat.units = 'degrees_north'
    load_cell_lat.long_name = 'loadcell_latitude'
    load_cell_lon.units = 'degrees_east'
    load_cell_lon.long_name = 'loadcell_longitude'
    # Assign data
    #  https://unidata.github.io/netcdf4-python/ (see "Dealing with Strings")
    #  sta_comp_id[:] = netCDF4.stringtochar(np.array(dmrows,dtype='S10'))
    #  load_cell_id[:] = netCDF4.stringtochar(np.array(load_cells,dtype='S10'))
    sta_comp_id._Encoding = 'ascii'
    sta_comp_id[:] = np.array(dmrows,dtype='S10')
    load_cell_id._Encoding = 'ascii'
    load_cell_id[:] = np.array(load_cells,dtype='S10')
    design_matrix[:,:] = desmat
    sta_comp_lat[:] = sclat
    sta_comp_lon[:] = sclon
    load_cell_lat[:] = lclat
    load_cell_lon[:] = lclon
    # Write Data to File
    dataset.close()

    # Read the netCDF file as a test
    f = netCDF4.Dataset(f_file)
    #print(f.variables)
    sta_comp_ids = f.variables['sta_comp_id'][:]
    load_cell_ids = f.variables['load_cell_id'][:]
    design_matrix = f.variables['design_matrix'][:]
    sta_comp_lat = f.variables['sta_comp_lat'][:]
    sta_comp_lon = f.variables['sta_comp_lon'][:]
    load_cell_lat = f.variables['load_cell_lat'][:]
    load_cell_lon = f.variables['load_cell_lon'][:]
    f.close()
    print(sta_comp_ids)
    print(design_matrix)

# --------------------- END CODE --------------------------- #


