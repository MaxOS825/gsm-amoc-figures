import sys
import os
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.basemap as mplt
from mpl_toolkits.basemap import Basemap
from datetime import datetime
from itertools import chain
from scipy import stats
from cdo import *

cwd = os.getcwd()

# time: last 10/20 years
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

exp=None
ref=None
if var != "sst" and var != "sictho" and var != "sicomo" and var != "wo" and var != "wmo":
    exp=np.ones(shape=(2,explen,12,40,180,360)) # ens,explen,time,depth,lat,lon
    ref=np.ones(shape=(2,explen,12,40,180,360))
elif var == "wo" or var == "wmo":
    exp=np.ones(shape=(2,explen,12,41,180,360)) # ens,explen,time,depth,lat,lon
    ref=np.ones(shape=(2,explen,12,41,180,360))
else:
    exp=np.ones(shape=(2,explen,12,180,360)) # ens,explen,time,depth,lat,lon
    ref=np.ones(shape=(2,explen,12,180,360))

cmap = ''
unit = ''
name = ''
extend = ''
clevs=[]
clevs_no0=[]
#depth = 0

if var == 'sao': # in your case rhoo, uho, sao, vke
    cmap = 'Blues'
    name = 'Sea water salinity'
    unit = '[psu]'
    extend='both'
    clevs=np.linspace(20,40,11)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 0

if var == 'rhoo': # in your case rhoo, uho, sao, vke
    cmap = 'Blues'
    name = 'Sea water density'
    unit = '[kg m-3]'
    extend='both'
    clevs=np.linspace(1025,1027.5,6)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 12

if var == 'uko': # in your case rhoo, uho, sao, vke; xvelocity
    cmap = 'YlGnBu_r'
    name = 'Sea water x velocity'
    unit = '[m s-1]'
    extend='both'
    clevs=np.linspace(-0.25,0.25,7)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 3

if var == 'vke': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'YlGnBu_r'
    name = 'Sea water y velocity'
    unit = '[m s-1]'
    extend='both'
    clevs=np.linspace(-10,10,20)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

if var == 'sst': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'Reds'
    name = 'Sea surface temperature'
    unit = '[K]'
    extend='both'
    clevs=np.linspace(265,305,21)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

if var == 'sictho': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'Blues'
    name = 'Sea ice thickness'
    unit = '[cm]'
    extend='both'
    clevs=np.linspace(0,500,11)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

if var == 'sicomo': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'coolwarm'
    name = 'Sea ice area fraction'
    unit = '[1]'
    extend='both'
    clevs=np.linspace(0.15,0.75,7)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

if var == 'wo': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'YlGnBu_r'
    name = 'Upward sea water velocity'
    unit = '[m s-1]'
    extend='both'
    clevs=np.linspace(0.25,1.75,7)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

if var == 'wmo': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'YlGnBu_r'
    name = 'Upward sea water transport'
    unit = '[kg s-1]'
    extend='both'
    clevs=np.linspace(0.25,1.75,7)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26
    
if var == 'umo': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'YlGnBu_r'
    name = 'Sea water x transport'
    unit = '[kg s-1]'
    extend='both'
    clevs=np.linspace(-0.25,0.25,7)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

if var == 'vmo': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'YlGnBu_r'
    name = 'Sea water y transport'
    unit = '[kg s-1]'
    extend='both'
    clevs=np.linspace(0.25,1.75,7)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

# load dataset, either SD or WD
files = []
#files2 = []
reffiles = []
#reffiles2 = []

for i in range(explen):
    files.append(cwd+'/'+str(scen)+'_ens'+str(1)+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231_remapped'+'.nc')
    files.append(cwd+'/'+str(scen)+'_ens'+str(2)+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231_remapped'+'.nc')
    reffiles.append(cwd+'/'+'REF'+'_ens'+str(1)+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231_remapped'+'.nc')
    reffiles.append(cwd+'/'+'REF'+'_ens'+str(2)+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231_remapped'+'.nc')

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

    f1 = netCDF4.Dataset(file,'r')
    f2 = netCDF4.Dataset(reffile, 'r')
    
    lat, latRef = f1.variables['lat'][:], f2.variables['lat'][:]
    lon, lonRef = f1.variables['lon'][:], f2.variables['lon'][:]
    depths = f1.variables['depth'][:]
    vargm = np.squeeze(f1.variables[var][:])
    vargm2 = np.squeeze(f2.variables[var][:])
    if ("ens1" in file):
        if var != "sst" and var != "sictho" and var != "sicomo":
            exp[0,ens1count,:,:,:,:] = vargm
        else:
            exp[0,ens1count,:,:,:] = vargm
        print(file + " done! Number ens1: ",ens1count)
        ens1count += 1
    if ("ens2" in file):
        if var != "sst" and var != "sictho" and var != "sicomo":
            exp[1,ens2count,:,:,:,:] = vargm
        else:
            exp[1,ens2count,:,:,:] = vargm
        print(file + " done! Number ens2: ",ens2count)
        ens2count += 1
    if ("ens1" in reffile):
        if var != "sst" and var != "sictho" and var != "sicomo":
            ref[0,refens1count,:,:,:,:] = vargm2
        else:
            ref[0,refens1count,:,:,:] = vargm2
        print(reffile + " done! Number REF ens1: ",refens1count)
        refens1count += 1
    if ("ens2" in reffile):
        if var != "sst" and var != "sictho" and var != "sicomo":
            ref[1,refens2count,:,:,:,:] = vargm2
        else:
            ref[1,refens2count,:,:,:] = vargm2
        print(reffile + " done! Number REF ens2: ",refens2count)
        refens2count += 1
    f1.close()
    f2.close()

if var != "sst" and var != "sictho" and var != "sicomo":
    ref_mean = np.mean(np.mean(np.mean(ref, axis=0), axis=0),axis=0)[depth]
else:
    ref_mean = np.mean(np.mean(np.mean(ref, axis=0), axis=0),axis=0)

if var == "uko" or var == "vke" or var == "wo" or var == "wmo":
    ref_mean = (np.ma.masked_where(ref_mean <= -10000,ref_mean))
else:
    ref_mean = (np.ma.masked_where(ref_mean <= 0,ref_mean))

fmt = '%i%%'

# actual data differences
if var == "sictho":
    ref_mean = ref_mean*1000 # convert to cm
topoin,lons2 = mplt.shiftgrid(180.,ref_mean,lon,start=False)

def all_files():

    fig, axes = plt.subplots(nrows=5, ncols=8,figsize=(11,8))
    fig.tight_layout()

    j = 0
    for ax in axes:
        for k in range(8):
            exp_mean = np.mean(np.mean(np.mean(exp, axis=0), axis=0),axis=0)[j]
            ref_mean = np.mean(np.mean(np.mean(ref, axis=0), axis=0),axis=0)[j]
            diff_rel = 100 * (exp_mean - ref_mean) / ref_mean # also 2D
            diff_abs = exp_mean - ref_mean # 2D
            topoin,lons2 = mplt.shiftgrid(180.,diff_abs,lon,start=False)
            topoin_rel,lons2 = mplt.shiftgrid(180.,diff_rel,lon,start=False)
            m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c',ax=ax[k])
            lon1, lat1 = np.meshgrid(lons2, lat)
            xi, yi = m(lon1, lat1)
            im1 = m.pcolormesh(xi,yi,topoin[:-1,:-1],shading='flat',cmap=cmap,latlon=True)
            cb = m.colorbar(im1,"bottom", size="5%", pad="2%")
            
            ax[k].set_title (var+"-"+scen+"-"+"depth" + str(j))
            ax[k].set_xlim([-100, 5])
            ax[k].set_ylim([7, 80])
            m.drawmapboundary(fill_color='0.3')
            m.drawcoastlines(linewidth=1.5)
            m.fillcontinents()
            print("Finished Nr:",j)
            j += 1

m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,\
    llcrnrlon=-180,urcrnrlon=180,resolution='c')


lon1, lat1 = np.meshgrid(lons2, lat)
xi, yi = m(lon1, lat1)

fig = plt.figure(figsize=(22, 8))

cs = m.contourf(xi,yi,topoin, clevs,cmap=cmap, extend='both')#,norm=norm), clevs

if var == "sictho":
    plt.colorbar(cs,label="cm")
else:
    plt.colorbar(cs,label=unit)

if var == "sao":
    plt.title ('REF: ' +name + ' at sea surface')
elif var == "sst" or var == "sictho" or var == "sicomo":
    plt.title ('REF: ' +name)
else:
    plt.title ('REF: ' +name + ' at depth: ' + str(depths[depth]) + 'm')


ax = plt.gca()
ax.set_xlim([8900000, 24000000])
ax.set_ylim([15000000, 27000000])
m.drawcoastlines(linewidth=1.5)
m.fillcontinents()

plt.savefig(path+expname+'_'+var+'_'+'REF'+'_'+str(depth)+'_'+year_begin+'_'+year_end+'.pdf',bbox_inches='tight')
#plt.close(fig)
print("PDF created!")






 