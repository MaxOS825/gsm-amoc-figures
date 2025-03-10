import sys
import os
import string
import statistics
import re
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.basemap as mplt
from mpl_toolkits.basemap import Basemap
from datetime import datetime
from itertools import chain
from scipy import stats
from cdo import *
from pathlib import Path

cdo = Cdo()
cdo.debug = True
cwd = os.getcwd()

# time: last 20 years
# depth: whatever shows data best

vars = sys.argv
var=vars[1] # sao, rhoo, uho, vke
year_begin=vars[2] # 2180
year_end=vars[3] # 2199
expname=vars[4] # mpiom_data_3du_mm
scen=vars[5] # REF, SD or WD
depth=int(vars[6])

explen=(int(year_end)-int(year_begin)+1)

path='plots/'+var+'/'
os.makedirs(path, exist_ok=True)

pvalue=0.05

exp=np.ones(shape=(2,explen,12,40,180,360)) # ens,explen,time,depth,lat,lon
ref=np.ones(shape=(2,explen,12,40,180,360))

cmap = ''
unit = ''
name = ''
extend = ''
clevs=[]
clevs_no0=[]
#depth = 0

if var == 'sao': # in your case rhoo, uho, sao, vke
    cmap = 'YlGnBu_r'
    name = 'Sea water salinity'
    unit = '[psu]'
    extend='both'
    clevs=np.linspace(-60,60,13)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 15

if var == 'rhoo': # in your case rhoo, uho, sao, vke
    cmap = 'YlGnBu_r'
    name = 'Sea water density'
    unit = '[kg m-3]'
    extend='both'
    clevs=np.linspace(0,40,80)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 12

if var == 'uko': # in your case rhoo, uho, sao, vke; xvelocity
    cmap = 'YlGnBu_r'
    name = 'Sea water x velocity'
    unit = '[m s-1]'
    extend='both'
    clevs=np.linspace(-60,60,13)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 25

if var == 'vke': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'YlGnBu_r'
    name = 'Sea water y velocity'
    unit = '[m s-1]'
    extend='both'
    clevs=np.linspace(-60,60,13)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

if var == 'tho': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'YlGnBu_r'
    name = 'Sea water pot. temp.'
    unit = '[C]'
    extend='both'
    clevs=np.linspace(-60,60,13)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

if var == 'sst': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'YlGnBu_r'
    name = 'Sea surface temp.'
    unit = '[K]'
    extend='both'
    clevs=np.linspace(-60,60,13)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

if var == 'sictho': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'YlGnBu_r'
    name = 'Sea ice thickness'
    unit = '[m]'
    extend='both'
    clevs=np.linspace(-60,60,13)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

if var == 'sicomo': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'YlGnBu_r'
    name = 'Sea ice area fraction'
    unit = '[1]'
    extend='both'
    clevs=np.linspace(-60,60,13)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26


# load dataset, either SD or WD
files = []
reffiles = []

for i in range(explen):
    files.append(cwd+'/'+str(scen)+'_ens'+str(1)+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231'+'.nc')
    files.append(cwd+'/'+str(scen)+'_ens'+str(2)+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231'+'.nc')
    reffiles.append(cwd+'/'+'REF'+'_ens'+str(1)+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231'+'.nc')
    reffiles.append(cwd+'/'+'REF'+'_ens'+str(2)+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231'+'.nc')

lat = None
lon = None
latRef = None
lonRef = None
vargm = None
vargm2 = None
ens1count = 0
ens2count = 0
refens1count = 0
refens2count = 0
for file, reffile in zip(files, reffiles):

    tempfile = Path(file.split('.')[0] + '_remapped.nc')
    f1 = netCDF4.Dataset(file.split('.')[0] + '_remapped.nc','r')
    f2 = netCDF4.Dataset(reffile.split('.')[0] + '_remapped.nc', 'r')
    
    lat, latRef = f1.variables['lat'][:], f2.variables['lat'][:]
    lon, lonRef = f1.variables['lon'][:], f2.variables['lon'][:]
    vargm = np.squeeze(f1.variables[var][:])
    vargm2 = np.squeeze(f2.variables[var][:])

    if ("ens1" in file):
        exp[0,ens1count,:,:,:,:] = vargm
        print(file + " done! Number ens1: ",ens1count)
        ens1count += 1
    if ("ens2" in file):
        exp[1,ens2count,:,:,:,:] = vargm
        print(file + " done! Number ens2: ",ens2count)
        ens2count += 1
    if ("ens1" in reffile):
        ref[0,refens1count,:,:,:,:] = vargm2
        print(reffile + " done! Number REF ens1: ",refens1count)
        refens1count += 1
    if ("ens2" in reffile):
        ref[1,refens2count,:,:,:,:] = vargm2
        print(reffile + " done! Number REF ens2: ",refens2count)
        refens2count += 1
    f1.close()
    f2.close()

def all_files_abs_diff():

    for j in range(40):
        exp_mean = np.mean(np.mean(np.mean(exp, axis=0), axis=0),axis=0)[j]
        ref_mean = np.mean(np.mean(np.mean(ref, axis=0), axis=0),axis=0)[j]

        diff_abs = exp_mean - ref_mean # 2D
        topoin,lons2 = mplt.shiftgrid(180.,diff_abs,lon,start=False)
        m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
        llcrnrlon=-180,urcrnrlon=180,resolution='c')
        lon1, lat1 = np.meshgrid(lons2, lat)
        xi, yi = m(lon1, lat1)
        cs = m.contourf(xi,yi, topoin, cmap=cmap, extend='both') #clevs
        plt.colorbar(cs)
        
        ax = plt.gca()
        ax.set_xlim([-100, 5])
        ax.set_ylim([7, 80])
        m.drawmapboundary(fill_color='0.3')
        m.drawcoastlines(linewidth=1.5)
        m.fillcontinents()
        print("Finished Nr:",j)
        plt.savefig(path+expname+'_'+var+'_'+'REF_'+scen+'_'+str(j)+'_'+year_begin+'_'+year_end+'_abs_diff.pdf',bbox_inches='tight')
        plt.close()
        print("PDF abs diff created for depth ",j)

all_files_abs_diff()

def all_files_abs():

    for j in range(40):
    
        ref_mean = np.mean(np.mean(np.mean(ref, axis=0), axis=0),axis=0)[j]
        
        topoin,lons2 = mplt.shiftgrid(180.,ref_mean,lon,start=False)
        
        m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
        llcrnrlon=-180,urcrnrlon=180,resolution='c')
        lon1, lat1 = np.meshgrid(lons2, lat)
        xi, yi = m(lon1, lat1)
        
        cs = m.contourf(xi,yi, topoin, cmap=cmap, extend='both') #clevs
        plt.colorbar(cs)
        
        ax = plt.gca()
        ax.set_xlim([-100, 5])
        ax.set_ylim([7, 80])
        m.drawmapboundary(fill_color='0.3')
        m.drawcoastlines(linewidth=1.5)
        m.fillcontinents()
        print("Finished Nr:",j)
        plt.savefig(path+expname+'_'+var+'_'+'REF_'+str(j)+'_'+year_begin+'_'+year_end+'.pdf',bbox_inches='tight')
        plt.close()
        print("PDF abs created for depth ",j)

all_files_abs()