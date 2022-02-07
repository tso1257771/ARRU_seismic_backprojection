#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 21:08:18 2021

@author: rick
"""
import os
from glob import glob
import numpy as np
import subprocess, collections
from obspy.geodetics import gps2dist_azimuth

ev_info = collections.namedtuple('ev_info','st stack_p lon lat dep')

BP_path = './out_data/BP_out'

time_dif = 2       # time threshold
dist_dif = 20      # dist threshold
stack_req = '2.0'  # stacking threshold

# start time
yr = 2019
jdy_s = 188
jdy_e = 188

#=====  get folder list
hr_list_glob = glob(os.path.join(BP_path, '????.???.??'))
hr_list = np.sort([os.path.basename(h) for h in hr_list_glob])
print('there are ', len(hr_list), ' folders!')  

for hr_dir in hr_list:
    print('working on : ', hr_dir)
    #===== go to the folder 
    os.chdir(os.path.join(os.path.abspath(BP_path), hr_dir))
    subprocess.call('cat 00*_BP.out > ' + hr_dir + '_BPev.list', shell=True)
    subprocess.call('cat 01*_BP.out >> ' + hr_dir + '_BPev.list', shell=True)
    subprocess.call('cat 02*_BP.out >> ' + hr_dir + '_BPev.list', shell=True)
    #==== select qualified stacking
    subprocess.call("awk '$2>" + stack_req + "{print $0}' " + hr_dir +\
        '_BPev.list > ' + hr_dir + '_BPev.list2', shell=True)
        
    #===== check file emepty or not
    if (os.stat(hr_dir + '_BPev.list2').st_size == 0):
        print(hr_dir + '_BPev.list2 is empty')
        continue
    
    #==== sort by time    
    subprocess.call('sort -n -o ' + hr_dir + '_BPev.list_sel '+hr_dir + '_BPev.list2', shell=True)
    
    #===== read all events
    print(hr_dir, ' read all events!')
    ev_file = open(hr_dir + '_BPev.list_sel', 'r')
    all_ev = []
    for i1, ev_tmp in enumerate(ev_file):
        ev_tmp2 = ev_tmp.strip().split()
        ev_st = float(ev_tmp2[0]) 
        ev_stack_p = float(ev_tmp2[1])
        ev_lon = float(ev_tmp2[2])
        ev_lat = float(ev_tmp2[3])
        ev_dep = float(ev_tmp2[4])
        all_ev.append(ev_info(ev_st,ev_stack_p,ev_lon,ev_lat,ev_dep))
    ev_file.close()
    print(hr_dir, i1+1, ' events!')
    
    #====== check events
    out_catalog =  '../' + hr_dir + '_BP_T' + str(time_dif) + 's_D' +\
        str(dist_dif)+'km.catalog'
    ev_out = open(out_catalog, 'w')
    rej_idx = []
    for i1, ev_tmp in enumerate(all_ev):
        ev_in = 1     # for event select or not
        if i1 in rej_idx:
            #print('skip ', i1)
            continue
        for i2 in range(i1+1, len(all_ev)):
            ev_dist, az, baz = gps2dist_azimuth(ev_tmp.lat, ev_tmp.lon, \
                all_ev[i2].lat, all_ev[i2].lon)
            ev_dist = np.sqrt((ev_dist / 1000)**2 + (all_ev[i2].dep-ev_tmp.dep)**2)  # convert to km
            tt_dif = all_ev[i2].st - ev_tmp.st
            #==== reject the event
            if ((tt_dif <= time_dif) & (ev_dist <= dist_dif)):
                if (ev_tmp.stack_p < all_ev[i2].stack_p): #=== reject master event
                    #print('reject1:',i1, i2, ev_tmp.st, all_ev[i2].st, ev_tmp.stack_p, \
                    #    all_ev[i2].stack_p, ev_dist)
                    ev_in = 0
                    break
                else: #==== reject checked event 
                    rej_idx.append(i2)
                    #print('rej checked: ',i2)
                                    
            #===== time diff too large break
            if tt_dif > time_dif:
                #print('time diff too large ', tt_dif)
                break
        #======= write or not
        if ev_in == 1:
            ev_out.write('{0.st:7.2f} {0.stack_p:7.4f} {0.lon:9.4f} {0.lat:8.4f} {0.dep:6.2f}\n'\
            .format(ev_tmp))        
    ev_out.close()
    