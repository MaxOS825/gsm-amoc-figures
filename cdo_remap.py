import sys
import os
from cdo import *

cdo = Cdo()
cdo.debug = True
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

# load dataset
files = []
reffiles = []
for i in range(explen):
    files.append(cwd+'/'+str(scen)+'_ens'+str(1)+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231'+'.nc')
    files.append(cwd+'/'+str(scen)+'_ens'+str(2)+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231'+'.nc')
    reffiles.append(cwd+'/'+'REF'+'_ens'+str(1)+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231'+'.nc')
    reffiles.append(cwd+'/'+'REF'+'_ens'+str(2)+'_'+expname+'_'+str(int(year_begin)+i)+'0101_'+str(int(year_begin)+i)+'1231'+'.nc')
    
for file, reffile in zip(files, reffiles):
    cdo.remapbil('r360x180',input=file, output=file.split('.')[0] + '_remapped.nc')
    print('Successfully remapped file: ' + file)
    cdo.remapbil('r360x180',input=reffile, output=reffile.split('.')[0] + '_remapped.nc')
    print('Successfully remapped reffile: ' + reffile)