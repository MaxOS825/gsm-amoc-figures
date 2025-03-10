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

# time: last 10 years
# depth: whatever shows data best

vars = sys.argv
var=vars[1] # sao, rhoo, uho, vke
year_begin=vars[2] # 2180
year_end=vars[3] # 2199
expname=vars[4] # mpiom_data_3du_mm
scen=vars[5] # REF, SD or WD
#depth=int(vars[6])

explen=(int(year_end)-int(year_begin)+1)

path='plots/'+var+'/'
os.makedirs(path, exist_ok=True)

pvalue=0.05

ref = None
ref2 = None
if var != "sst" and var != "sictho" and var != "sicomo" and var != "wo" and var != "wmo":
    ref=np.ones(shape=(2,explen,12,40,180,360))
    if var == "vmo" or var == "umo":
        ref2 = np.ones(shape=(2,explen,12,40,180,360))
elif var == "wo" or var == "wmo":
    ref=np.ones(shape=(2,explen,12,41,180,360))
else:
    ref=np.ones(shape=(2,explen,12,180,360))

cmap = ''
cmap2 = 'YlGnBu'
unit = ''
unit2 = '[psu]'
name = ''
extend = ''
clevs=[]
clevs2=np.linspace(20,40,11)
clevs_no0=[]

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
    #clevs=np.linspace(1025,1027.5,11)
    clevs=np.linspace(1026,1032,7)
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
    cmap = 'Blues'
    name = 'Sea surface temperature'
    unit = '[K]'
    extend='both'
    clevs=np.linspace(275,305,15)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

if var == 'sictho': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'YlGnBu'
    name = 'Sea ice thickness'
    unit = '[m]'
    extend='both'
    clevs=np.linspace(0.2,0.5,5)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

if var == 'sicomo': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'YlGnBu'
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
    cmap = 'Blues'
    name = 'Sea water x and y transport' #x
    unit = '[10^15 kg s-1]'
    extend='both'
    clevs=np.linspace(0,150,51)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26

if var == 'vmo': # in your case rhoo, uho, sao, vke; yvelocity
    cmap = 'Blues'
    name = 'Sea water x and y transport' #y
    unit = '[10^15 kg s-1]'
    extend='both'
    clevs=np.linspace(0,10,9)
    clevs_no0=[-30,-25,-20,-15,-10,-5,5,10,15,20,25,30]
    #depth = 26


# load dataset, either SD or WD
files = []
reffiles = []

for i in range(explen):
    files.append(cwd+'/'+str(scen)+'_ens'+str(1)+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231_remapped'+'.nc')
    files.append(cwd+'/'+str(scen)+'_ens'+str(2)+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231_remapped'+'.nc')
    reffiles.append(cwd+'/'+'REF'+'_ens'+str(1)+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231_remapped'+'.nc')
    reffiles.append(cwd+'/'+'REF'+'_ens'+str(2)+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231_remapped'+'.nc')

lat = None
lon = None
vargm2 = None
vargm3 = None
refens1count = 0
refens2count = 0
for file, reffile in zip(files, reffiles):

    # load files
    f2 = netCDF4.Dataset(reffile, 'r')
    
    lat = f2.variables['lat'][:]
    lon = f2.variables['lon'][:]
    depths = f2.variables['depth'][:]
    vargm2 = np.squeeze(f2.variables[var][:])
    if var == "umo":
        vargm3 = np.squeeze(f2.variables["vmo"][:])
    if var == "vmo":
        vargm3 = np.squeeze(f2.variables["umo"][:])
    if ("ens1" in reffile):
        if var != "sst" and var != "sictho" and var != "sicomo":
            ref[0,refens1count,:,:,:,:] = vargm2
            if var == "vmo" or var == "umo":
                ref2[0,refens1count,:,:,:,:] = vargm3
        else:
            ref[0,refens1count,:,:,:] = vargm2
        print(reffile + " done! Number REF ens1: ",refens1count)
        refens1count += 1
    if ("ens2" in reffile):
        if var != "sst" and var != "sictho" and var != "sicomo":
            ref[1,refens2count,:,:,:,:] = vargm2
            if var == "vmo" or var == "umo":
                ref2[1,refens2count,:,:,:,:] = vargm3
        else:
            ref[1,refens2count,:,:,:] = vargm2
        print(reffile + " done! Number REF ens2: ",refens2count)
        refens2count += 1
    f2.close()

def all_files():
    levels = None
    if var in ["sst","sictho","sicomo"]:
        levels = 1
    elif var in ["wmo","wo"]:
        levels = 7 #41
    else:
        levels = 7 #40

    valList = []
    for j in range(levels):
        if var != "sst" and var != "sictho" and var != "sicomo":
            ref_mean = np.mean(np.mean(np.mean(ref, axis=0), axis=0),axis=0)[j+17]
            if var == "umo" or var == "vmo":
                ref_mean2 = np.mean(np.mean(np.mean(ref2, axis=0), axis=0),axis=0)[j+17]
        else:
            ref_mean = np.mean(np.mean(np.mean(ref, axis=0), axis=0),axis=0)

        if var == "uko" or var == "vke" or var == "wo" or var == "wmo" or var == "vmo" or var == "umo":
            ref_mean = (np.ma.masked_where(ref_mean <= -9999999999999,ref_mean))
            ref_mean = (np.ma.masked_where(ref_mean >= 9999999999999,ref_mean))
            if var == "umo" or var == "vmo":
                ref_mean2 = (np.ma.masked_where(ref_mean2 <= -9999999999999,ref_mean2))
                ref_mean2 = (np.ma.masked_where(ref_mean2 >= 9999999999999,ref_mean2))
        else:
            ref_mean = (np.ma.masked_where(ref_mean <= 0,ref_mean))

        ref_mean = np.nan_to_num(ref_mean,nan=0.99)
        if var == "umo" or var == "vmo":
            ref_mean2 = np.nan_to_num(ref_mean2,nan=0.99)
            #ref_mean3 = np.nan_to_num(ref_mean3,nan=0.99)
        topoin,lons2 = mplt.shiftgrid(180.,ref_mean,lon,start=False)

        if var == "umo" or var == "vmo":
            topoin2,lons3 = mplt.shiftgrid(180.,ref_mean2,lon,start=False)
            #topoin3,lons4= mplt.shiftgrid(180.,ref_mean3,lon,start=False)

        xval = []
        yval = []
        for k in range(len(lat)):
            for l in range(len(lon)):
                if lat[k] >= 7 and lat[k] <= 80 and lon[l] >= -100 and lon[l] <= 5:
                    xval.append(k) #lat[k]
                    yval.append(l) #lon[l]

        tempVal = np.ones(shape=(len(xval),len(yval)))
        tempVal = np.nan_to_num(tempVal,nan=0.99)
        for k in range(len(xval)):
            for l in range(len(yval)):
                tempVal[k,l] = ref_mean[xval[k],yval[l]]

        if abs(np.nanmax(tempVal)) <= abs(np.nanmin(tempVal)):
            val = abs(np.nanmin(tempVal))
        else:
            val = abs(np.nanmax(tempVal))
        valList.append(val)
        if np.isnan(valList[j]):
            val = valList[j-1]
        file = path+var+'_'+'REFabs'+'_'+str(explen)+'_'+str(j+17)+'.txt'
        if not os.path.exists(file) and not np.isnan(val):
            f = open(file, "w")
            f.write(str(val))
            f.close()
        elif not os.path.exists(file) and np.isnan(val):
            f = open(file, "w")
            f.write(str(valList[j-1]))
            f.close()

        f = open(file, "r")
        val = float(f.read())
        if np.isnan(val):
            val = valList[j-1+17]

        m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,\
        llcrnrlon=-180,urcrnrlon=180,lon_0=0.,resolution='c') #lat_1=20.,lat_2=40.,
        lon1, lat1 = np.meshgrid(lons2, lat)
        xi, yi = m(lon1, lat1)
        lon2, lat2 = np.meshgrid(lons3, lat)
        xi2, yi2 = m(lon2, lat2)


        # plotting
        fig = plt.figure(figsize=(22, 8))

        if var == "umo" or var == "vmo":
            # Directional vectors for umo and vmo
            gradx,grady=np.gradient(topoin*topoin2)
            gradmag=np.sqrt(gradx**2+grady**2)
            gradmag = gradmag/1000000000000000 # 10^15
            gradxnorm=gradx/gradmag
            gradynorm=grady/gradmag
            step=3
            
            u = topoin
            v = topoin2

            if var != "umo" and var != "vmo":
               cs = m.contourf(xi,yi,topoin,clevs,cmap=cmap, extend='both')#,norm=norm), clevs
            else:
               cs = m.contourf(xi,yi,gradmag,clevs,cmap=cmap, extend='both')#,norm=norm), clevs
            
            # Plotting Vector Field with QUIVER 
            qv1 = m.quiver(xi[::step,::step], yi[::step,::step], u[::step,::step], v[::step,::step], color='black',units='width', scale=10000000000)

        
            if var == "sicomo":
                plt.colorbar(cs,label="%")
            elif var == "umo" or var == "vmo":
                plt.colorbar(cs,label=unit)
            

        if var == "sao":
            plt.title ('REF: ' +name + ' at sea surface')
        elif var == "sst" or var == "sictho" or var == "sicomo":
            plt.title ('REF: ' +name)
        else:
            plt.title ('REF: ' +name + ' at depth: ' + str(depths[j+17]) + 'm')

        ax = plt.gca()
        ax.set_xlim([8900000, 24000000])
        ax.set_ylim([15000000, 27000000])
        m.drawcoastlines(linewidth=1.5)
        m.fillcontinents()
        #plt.show()
        plt.savefig(path+expname+'_'+var+'_'+'REF_'+str(17+j)+'_'+year_begin+'_'+year_end+'.pdf',bbox_inches='tight')
        print("Finished depth level: ",17+j)
        plt.close()

all_files()