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

cwd = os.getcwd()

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
    cmap = 'coolwarm'
    name = 'Sea water salinity'
    unit = '[psu]'
    extend='both'
    clevs=np.linspace(-1.5,1.5,9)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 0

if var == 'rhoo': # in your case rhoo, uho, sao, vke
    cmap = 'coolwarm'
    name = 'Sea water density'
    unit = '[kg m-3]'
    extend='both'
    clevs=np.linspace(1025,1027.5,11)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 12

if var == 'uko': # in your case rhoo, uho, sao, vke; xvelocity
    cmap = 'YlGnBu_r'
    name = 'Sea water x velocity'
    unit = '[m s-1]'
    extend='both'
    clevs=np.linspace(0.05,0.2,8)
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
    cmap = 'coolwarm'
    name = 'Sea surface temperature'
    unit = '[K]'
    extend='both'
    clevs=np.linspace(-1,1,11)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

if var == 'sictho': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'coolwarm'
    name = 'Sea ice thickness'
    #unit = '[m]'
    unit = '[%]'
    extend='both'
    #clevs=np.linspace(-100,100,11)
    clevs=np.linspace(3000,9000,11)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

if var == 'sicomo': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'coolwarm'
    name = 'Sea ice area fraction'
    unit = '[1]'
    extend='both'
    clevs=np.linspace(-100,100,51)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26


if var == 'wo': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'bwr'
    name = 'Upward sea water transport'
    unit = '[kg s-1]'
    extend='both'
    clevs=np.linspace(-100,100,21)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26
    
if var == 'umo': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'bwr'
    name = 'Sea water x and y transport' #x
    unit = '[%]'
    extend='both'
    clevs=np.linspace(-100,100,21)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

if var == 'vmo': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'bwr'
    name = 'Sea water x and y transport' #y
    unit = '[%]'
    extend='both'
    clevs=np.linspace(-100,100,21)
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
    #files.append('/Users/maximilian/Documents/Environmental Technology and International Affairs/data_sample2'+'/'+scen+'_ens'+refname+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231'+'.nc')
    #reffiles.append('/Users/maximilian/Documents/Environmental Technology and International Affairs/data_sample2'+'/'+'REF'+'_ens'+refname+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231'+'.nc')

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

    # load files
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


if var != "sst" and var != "sictho" and var != "sicomo" and var != "wo" and var != "wmo":
    ref_mean = np.mean(np.mean(np.mean(ref, axis=0), axis=0),axis=0)[depth]
    exp_mean = np.mean(np.mean(np.mean(exp, axis=0), axis=0),axis=0)[depth]
elif var == "wo" or var == "wmo":
    ref_mean = (np.reshape(ref,(explen*12*2,41,180,360))) # years * months * ensembles
    exp_mean = (np.reshape(exp,(explen*12*2,41,180,360)))
else:
    ref_mean = np.mean(np.reshape(ref,(explen*12*2,180,360)),axis=0) # years * months * ensembles
    exp_mean = np.mean(np.reshape(exp,(explen*12*2,180,360)),axis=0)

diff_rel = 100 * (exp_mean - ref_mean) / ref_mean # also 2D
fmt = '%i%%'

print("Means done!")

ref_time = None
exp_time = None
if var != "sst" and var != "sictho" and var != "sicomo" and var != "wo" and var != "wmo":
    ref_time = (np.reshape(ref,(explen*12*2,40,180,360))) # years * months * ensembles
    exp_time = (np.reshape(exp,(explen*12*2,40,180,360)))
elif var == "wo" or var == "wmo":
    ref_time = (np.reshape(ref,(explen*12*2,41,180,360))) # years * months * ensembles
    exp_time = (np.reshape(exp,(explen*12*2,41,180,360)))
else:
    ref_time = (np.reshape(ref,(explen*12*2,180,360))) # years * months * ensembles
    exp_time = (np.reshape(exp,(explen*12*2,180,360)))

# t-test significance - depth as significance!
sign = None
if var in ["sst", "sictho","sicomo"]:
    sign = (np.array(stats.ttest_ind(exp_time, ref_time, axis=0, equal_var=True))[1,...])
elif var in ["wo","wmo"]:
    sign = (np.array(stats.ttest_ind(exp_time[:,depth,:,:], ref_time[:,depth,:,:], axis=0, equal_var=True))[1,...])
else:
    sign = (np.array(stats.ttest_ind(exp_time[:,depth,:,:], ref_time[:,depth,:,:], axis=0, equal_var=True))[1,...])

sign = np.nan_to_num(sign,nan=0.99)

print("T-test done!")


# actual data differences
topoin_rel,lons2 = mplt.shiftgrid(180.,diff_rel,lon,start=False)
topoin2,lons2 = mplt.shiftgrid(180.,sign,lon,start=False)

m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,\
    llcrnrlon=-180,urcrnrlon=180,resolution='c')


lon1, lat1 = np.meshgrid(lons2, lat)
xi, yi = m(lon1, lat1)

fig = plt.figure(figsize=(22, 8))

cs = m.contourf(xi,yi,topoin_rel,clevs,cmap=cmap, extend='both')#,norm=norm), clevs
plt.colorbar(cs,label="%")
# significance shading
cs2 = m.contourf(xi,yi,topoin2,[pvalue,1],cmap=plt.get_cmap("gray"),hatches=['..',None],alpha=pvalue) #plt.get_cmap("gray")
plt.clabel(cs2, inline = True, fontsize=8, fmt='%d', colors = 'black')

# significance shading
#im2 = m.pcolormesh(xi,yi,topoin2[:-1,:-1],shading='flat',cmap=plt.cm.gist_gray,latlon=True)
#plt.colorbar(im1)
if var == "sao":
    plt.title (scen + '-REF rel diff: ' +name + ' at sea surface')
elif var == "sst" or var == "sictho" or var == "sicomo":
    plt.title (scen + '-REF rel diff: ' +name)
else:
    plt.title (scen + '-REF rel diff: ' +name + ' at depth: ' + str(depths[depth]) + 'm')
#plt.title ('REF abs: ' +name + ' at depth: ' + str(depths[depth]) + 'm')
#plt.xlabel('Longitude'    + ' (' + 'degrees_east'    + ')')
#plt.ylabel('Latitude'    + ' (' + 'degrees_north'    + ')')

#m.drawparallels(np.arange(-90.,99.,30.))
#m.drawmeridians(np.arange(-180.,180.,60.))
ax = plt.gca()
ax.set_xlim([8900000, 24000000])
ax.set_ylim([15000000, 27000000])
#m.drawmapboundary(fill_color='0.3')
m.drawcoastlines(linewidth=1.5) 
m.fillcontinents()



# save the pdf
#plt.show()
plt.savefig(path+expname+'_'+var+'_'+scen+'-REF_'+str(depth)+'_'+year_begin+'_'+year_end+'_rel_diff.pdf',bbox_inches='tight')
#plt.close(fig)
print("PDF created!")