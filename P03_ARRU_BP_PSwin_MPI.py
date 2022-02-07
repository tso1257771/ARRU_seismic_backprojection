#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:23:48 2021

@author: rick
"""
import collections
import os
import struct
import time
import numpy as np
from mpi4py import MPI
from numba import jit


@jit(nopython=True)
def stacking_p_s(tt_shift, wf_P, wf_S, i1, dt, bp_len, bp_stack):

    idx_Pp = int(np.round(tt_shift[i1][1] / dt)) - 1
    idx_Sp = int(np.round(tt_shift[i1][2] / dt)) - 1

    bp_stack = bp_stack + wf_P[idx_Pp:idx_Pp + bp_len]
    bp_stack = bp_stack + wf_S[idx_Sp:idx_Sp + bp_len]

    return bp_stack


# check MPI env
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('|    rank', rank, 'of size', size)

# named tuple for storing station information
sta_i = collections.namedtuple("sta_info", "net sta chan loc")
grid_i = collections.namedtuple("grid_info", "lon lat dep id")

t0 = time.time()

grid_f = './metadata/SC_grid.txt'
sta_sel = './metadata/SC_sta_sel.list'
tt_path = './metadata/TTtable_25sta/'
wf_path = './out_data/ARRU_pred_bin/'
out_path = './out_data/BP_out/'

sta_list = open(sta_sel, 'r')
grid_list = open(grid_f, 'r')

# ==== parameters
tt_sta = 25  # number of stations in tt list
bp_sta = 8  # number of stations used for BP
dt = 0.01  # delta of waveforms
PS_range = 0.5  # P-S differences +/- PS_range between calculated & observed
sel_prob = 1.5  # value for potential event selection
ev_orig_dif = round( 1.0 / dt)  # origin time difference requirements

wf_len = int(np.round((60 * 60 + 60) / dt))  # length of WFs for back projection
bp_len = int(np.round(60 * 60 / dt))  # stack length

# ==== stack
zero_front = 1  # add zeros before P
zero_back = 1  # add zeros after S

b_size = 4

# start time
yr = 2019
jdy_s = 188
jdy_e = 188

hr_s = 8
hr_e = 8

# ===== store stations
sta_all = []
for i1, li1 in enumerate(sta_list):
    tmp = li1.strip().split()
    # ==== store net, sta, chan, loc
    sta_all.append(sta_i(tmp[0], tmp[1], tmp[2], tmp[3]))
sta_list.close()
sta_num = i1 + 1  # number of total stations
print('There are ', sta_num, ' stations!')

# ==== store grid info
grid_all = []
for i1, li1 in enumerate(grid_list):
    tmp = li1.strip().split()
    # === store: lon, lat, dep, id
    grid_all.append(grid_i(float(tmp[0]), float(tmp[1]), float(tmp[2]), tmp[3]))
grid_list.close()
grid_num = i1 + 1  # number of total grids
print('There are ', grid_num, ' grids!')

# ====== store grid : staID, Pt, St
print('read all grid tt for stations!')
tt_shift = np.zeros((grid_num, tt_sta * 3), dtype='float32')
for i1, igrid in enumerate(grid_all):
    tt_f = open(tt_path + igrid.id + '_'+str(tt_sta)+'_noW', 'r')
    for i2, txt2 in enumerate(tt_f):
        tmp = txt2.strip().split()
        for i3, txt3 in enumerate(tmp):
            tt_shift[i1][i3] = float(tmp[i3])
    tt_f.close()
print('finish reading grid time table!')

# ======== start backprojection
print('start backprojection!')
for idy in range(jdy_s, jdy_e + 1):
    for ihr in range(hr_s, hr_e + 1):
        t0 = time.time()
        hr_mk = str(yr) + '.' + str(idy).zfill(3) + '.' + str(ihr).zfill(2)
        #print('work on ', hr_mk)
        out_dir = out_path + hr_mk + '/'
        if rank == 0:
            if not os.path.exists(out_dir):
                print(rank, ' create folder : ', hr_mk)
                os.makedirs(out_dir)
        comm.Barrier()
        print(rank, 'work on ', hr_mk)
        # ===== read all the station waveforms
        print('start reading P & S of ', hr_mk)
        sta_check = np.zeros(sta_num, dtype='int8')  # for checking
        sta_Pwf = np.zeros((sta_num, wf_len), dtype='float32')
        sta_Swf = np.zeros((sta_num, wf_len), dtype='float32')
        for i1, ista in enumerate(sta_all):
            P_fnam = wf_path + hr_mk + '/' + ista.net + '.' + ista.sta + '.' + ista.chan + '.' + hr_mk + '.P.bin'
            S_fnam = wf_path + hr_mk + '/' + ista.net + '.' + ista.sta + '.' + ista.chan + '.' + hr_mk + '.S.bin'
            if not (os.path.exists(P_fnam)):
                print(P_fnam, ' is missing!')
                continue
            if not (os.path.exists(S_fnam)):
                print(S_fnam, ' is missing!')
                continue
            # =======
            sta_check[i1] = 1  # station is in
            P_f = open(P_fnam, 'rb')
            S_f = open(S_fnam, 'rb')
            sta_Pwf[i1][:] = struct.unpack('f' * wf_len, P_f.read(b_size * wf_len))
            sta_Swf[i1][:] = struct.unpack('f' * wf_len, S_f.read(b_size * wf_len))
            P_f.close()
            S_f.close()
        # ======== finish reading all prob of all stations
        print('finish reading P & S of ', hr_mk)
        # ======== do the stacking of each grid

        # ======== MPI start
        for i2 in range(rank, len(grid_all), size):
            igrid = grid_all[i2]
            bp_stack = np.zeros((1, bp_len), dtype='float32')  # store all backprojection results
            st_pick = 0  # number of station picked
            for i3 in range(tt_sta):
                sta_idx = int(tt_shift[i2][i3 * 3]) - 1  # the number is start from 1 ; - 1 for python
                # ==== if station is not exist
                if sta_check[sta_idx] < 1:
                    continue
                # ====== temp waveforms for stacking
                st_pick += 1
                Pwf_tmp = np.zeros((1, wf_len), dtype='float32')
                Swf_tmp = np.zeros((1, wf_len), dtype='float32')
                Pwf_tmp = sta_Pwf[sta_idx][:].copy()
                Swf_tmp = sta_Swf[sta_idx][:].copy()
                # ===== if there are P-S pairs file reject unqualified pairs
                PS_fnam = wf_path + hr_mk + '/' + sta_all[sta_idx].net + '.' + sta_all[sta_idx].sta \
                          + '.' + sta_all[sta_idx].chan + '.' + hr_mk + '.win'
                cal_PS = (tt_shift[i2][i3 * 3 + 2] - tt_shift[i2][i3 * 3 + 1])  # S-P time diff from time table
                if os.path.exists(PS_fnam):
                    PS_f = open(PS_fnam, 'r')
                    for i4, li4 in enumerate(PS_f):
                        tmp = li4.strip().split()
                        obs_PS = (int(tmp[1]) - int(tmp[0])) * dt  # observed P-S window time
                        # print(sta_all[st_idx].sta,i2,'OBS PS',obs_PS,' cal PS',cal_PS)
                        if np.abs(cal_PS - obs_PS) > PS_range:  # time difference larger than threshold
                            # print(sta_all[st_idx].sta,' PS rej ',i2)
                            sp0 = int(tmp[0]) - 1  # [py index] index - 1 for python
                            ep0 = int(tmp[1]) - 1  # [py index] index - 1 for python
                            if sp0 < zero_front:
                                Pwf_tmp[0:ep0 + zero_back] = 0  # [py index]
                                Swf_tmp[0:ep0 + zero_back] = 0  # [py index]
                            elif (ep0 + zero_back) >= wf_len:
                                Pwf_tmp[sp0 - zero_front:wf_len] = 0  # [py index]
                                Swf_tmp[sp0 - zero_front:wf_len] = 0  # [py index]
                            else:
                                Pwf_tmp[sp0 - zero_front:ep0 + zero_back] = 0  # [py index]
                                Swf_tmp[sp0 - zero_front:ep0 + zero_back] = 0  # [py index]
                    PS_f.close()
                # *********************************************************
                # ======== stacking P and S probs
                idx_Pp = int(np.round(tt_shift[i2][i3 * 3 + 1] / dt)) - 1
                idx_Sp = int(np.round(tt_shift[i2][i3 * 3 + 2] / dt)) - 1
                # print('sta & P & S shift:', sta_all[st_idx].sta,idx_Pp,idx_Sp)
                bp_stack = bp_stack + Pwf_tmp[idx_Pp:idx_Pp + bp_len]
                bp_stack = bp_stack + Swf_tmp[idx_Sp:idx_Sp + bp_len]
                # ***************************************************************
                # bp_stack = stacking_p_s(tt_shift, wf_P, wf_S, i1, dt, bp_len, bp_stack)
                # ***************************************************************
                # ===== if number of station == required
                if st_pick == bp_sta:
                    break

            # ======== if the stacking values larger than thershold print out
            if np.amax(bp_stack[0][:]) <= sel_prob:
                #print(rank,hr_mk, ' ', igrid.id, ' no events!')
                pass
            else:  # write out qualified values
                print(rank,hr_mk, ' ', igrid.id, ' write events!')
                out_f = open(out_dir + igrid.id + '_BP.out', 'w')
                ev_idx = np.argwhere(bp_stack[0][:] > sel_prob)
                sel_idx = ev_idx[0][0]  # 1st quilified index
                sel_stackV = bp_stack[0][sel_idx]  # 1st qualified value
                for i5, idx2 in enumerate(ev_idx):
                    # ====== origin time larger than required : write out & replace the new one
                    if (np.abs(idx2[0] - sel_idx) > ev_orig_dif) | (i5 == (len(ev_idx) - 1)):
                        out_f.write('{0:8.2f} {1:6.4f} {2:9.4f} {3:8.4f} {4:6.2f}\n'.format( \
                            (sel_idx + 1) * dt, sel_stackV, igrid.lon, igrid.lat, igrid.dep))
                        # print(hr_mk,' ',ig.id,' replace1 ',sel_idx,sel_stackV,' --> ',idx2[0],bp_stack[idx2[0]])
                        sel_idx = idx2[0]
                        sel_stackV = bp_stack[0][sel_idx]
                    # ====== if diff in the range & has larger value ==> replace
                    if sel_stackV < bp_stack[0][idx2[0]]:
                        # print(hr_mk,' ',ig.id,' replace2 ',sel_idx,sel_stackV,' --> ',idx2[0],bp_stack[idx2[0]])
                        sel_idx = idx2[0]
                        sel_stackV = bp_stack[0][sel_idx]
                out_f.close()
        # ======== MPI  end
        t1 = time.time()
        print('Time for ', hr_mk, ' = ', t1 - t0, ' Secs')
        #comm.Barrier()

comm.Barrier()
MPI.Finalize()
print('Mission Completed!')
