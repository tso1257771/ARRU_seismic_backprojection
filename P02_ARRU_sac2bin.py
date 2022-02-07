#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:44:16 2021

(1) convert ARRU sac to binary file
(2) store P&S pairs 

@author: rick
"""
import os
import numpy as np
from glob import glob
from obspy import  read, UTCDateTime
from obspy.signal.trigger import trigger_onset

ARRU_path = './out_data/ARRU_pred'
out_path = './out_data/ARRU_pred_bin'
sta_sel = './metadata/SC_sta_sel.list'

#=== select criteria
Pp_req = 0.3
Sp_req = 0.3
pair_req1 = 0.3 #ative thershold
pair_req2 = 0.3 #deactive thershold
PS_min = 0.5 # min P-S pair length

wf_delta = 0.01
wf_len = int(np.round( (60 * 60 + 60) / wf_delta))  # length of decimate len


#==== read stations info
print('read station files!')
sta_all=[]
sta_list = open(sta_sel,'r')
for i1,line in enumerate(sta_list):
    info = line.strip().split()
    sta_all.append(info[0]+'.'+info[1]+'.'+info[2])
sta_list.close() 
print(len(sta_all),' stations!') 
sta_tmp = set(sta_all)
sta_all = list(sta_tmp)
print(len(sta_all),' unique stations!') 

#===== loop through hours
dir_idx = np.sort(glob(os.path.join(ARRU_path, '????.???.??')))
for d in range(len(dir_idx)):
    print(f"Processing: {d+1}/{len(dir_idx)}: {dir_idx[d]}")
    current_hr = os.path.basename(dir_idx[d])
    yr, jday, ihr = current_hr.split('.')
    #for ihr in range(8,9):
    #current_hr = yr + '.' + jdy + '.' + str(ihr).zfill(2)
    
    #===== create folder for hour sac
    data_dir = os.path.join(ARRU_path, current_hr)
    if not os.path.exists(data_dir):
        print(current_hr, ' is missing!')
        continue
    #===================
    hr_ot = UTCDateTime(yr+'-'+jdy+'T'+str(ihr)+':00:00.0')
    out_dir = os.path.join(out_path, current_hr)
    #==== check & create folder
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #=== loop through station waveforms
    for ista in sta_all:
        sta_mk =  ista + '.' + current_hr
        P_nam = glob(os.path.join(data_dir, ista + '.*.sac.P'))
        S_nam = glob(os.path.join(data_dir, ista + '.*.sac.S'))
        EQ_nam = glob(os.path.join(data_dir, ista + '.*.sac.mask'))
        print('working on :', current_hr, ista)
        #=== skip missing data
        if len(P_nam) == 0:
            print('missing '+ os.path.join(data_dir, ista + '.*.sac.P'))
            continue
        if len(S_nam) == 0:
            print('missing '+ os.path.join(data_dir, ista + '.*.sac.S'))
            continue
        if len(EQ_nam) == 0:
            print('missing '+ os.path.join(data_dir, ista + '.*.sac.mask'))
            continue
        
        #====== P & S both exist
        P_wf = read(P_nam[0])
        S_wf = read(S_nam[0])
        EQ_wf = read(EQ_nam[0])
        #===== check delta
        if np.fabs(P_wf[0].stats.delta-wf_delta) > 0.001:
            print('waveform delta problem!', current_hr, ista)
            exit
        #===== check nan in array
        #P_nan = np.isnan(P_wf[0].data)
        #S_nan = np.isnan(S_wf[0].data)            
        #if (P_nan.any()|S_nan.any()):
        #    print('WFs has Nan!',current_hr, ista)
        #    exit
        #======  for P, S, and EQ mask
        if P_wf[0].stats.starttime > hr_ot:
            print('start time > hour origin time!')
            exit
        #==== find the start and end indexes 
        Pidx_s = round((hr_ot - P_wf[0].stats.starttime)/P_wf[0].stats.delta)
        Pidx_e = Pidx_s + wf_len 
        Psel_wf = np.zeros(wf_len, dtype='float32')    
        Psel_wf = np.float32(P_wf[0].data[Pidx_s:Pidx_e])
        
        #===== for S
        if S_wf[0].stats.starttime > hr_ot:
            print('start time > hour origin time!')
            exit
        #==== find the start and end indexes 
        Sidx_s = round((hr_ot - S_wf[0].stats.starttime)/S_wf[0].stats.delta)
        Sidx_e = Sidx_s + wf_len 
        Ssel_wf = np.zeros(wf_len, dtype='float32')    
        Ssel_wf = np.float32(S_wf[0].data[Sidx_s:Sidx_e])
        
        #===== for EQ mask
        if EQ_wf[0].stats.starttime > hr_ot:
            print('start time > hour origin time!')
            exit
        #==== find the start and end indexes 
        EQidx_s = round((hr_ot - EQ_wf[0].stats.starttime)/EQ_wf[0].stats.delta)
        EQidx_e = EQidx_s + wf_len
        EQsel_wf = np.zeros(wf_len, dtype='float32')    
        EQsel_wf = np.float32(EQ_wf[0].data[EQidx_s:EQidx_e])
        
        #==== 
        Pout_nam = os.path.join(out_dir, sta_mk+'.P.bin')
        Sout_nam = os.path.join(out_dir, sta_mk+'.S.bin')
        EQout_nam = os.path.join(out_dir, sta_mk+'.win')
        #=== write out P & S to binary files
        Psel_wf.astype('float32').tofile(Pout_nam)
        Ssel_wf.astype('float32').tofile(Sout_nam)  
            
        #======== check P-S pair win
        pair_list = trigger_onset(EQsel_wf, pair_req1, pair_req2)
        #=====
        if len(pair_list) < 1:
            print('no P-S pairs')
            continue
            
        pair_count = 0 
        for pair_idx in pair_list:
            pair_sp =  pair_idx[0] - 1
            if pair_sp < 0:
                pair_sp=0
            pair_ep = pair_idx[1] + 1
            if pair_ep > len(EQsel_wf):
                pair_ep = len(EQsel_wf)
                
            #===== if P-S length too short
            if (((pair_ep-pair_sp)*wf_delta)<PS_min):                    
                continue                    
            #=======
            P_max_idx = np.argmax(Psel_wf[pair_sp:pair_ep]) # P max index 
            P_max_val = np.amax(Psel_wf[pair_sp:pair_ep])  # P max prob.
            S_max_idx = np.argmax(Ssel_wf[pair_sp:pair_ep]) # S max index 
            S_max_val = np.amax(Ssel_wf[pair_sp:pair_ep])   # S max prob.
            if (P_max_idx < S_max_idx) and (P_max_val > Pp_req) and (S_max_val > Sp_req):
                if pair_count == 0:
                    EQout_f = open(EQout_nam,'w')
                pair_count += 1
                EQout_f.write('{:7d} {:7d}\n'.format(pair_sp,pair_ep))                    
        if pair_count > 0:
            EQout_f.close()                
