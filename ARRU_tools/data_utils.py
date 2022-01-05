import gc
import numpy as np
from obspy import read
from obspy.signal.trigger import trigger_onset
from copy import deepcopy

def conti_standard_wf_fast(wf, pred_npts, pred_interval_sec, dt, 
        pad_zeros=True):
    '''
    input: 
    wf: obspy.stream object (raw_data)
    pred_npts
    pred_interval_sec
    pad_zeros: pad zeros before after the waveform for full repeating predictions 

    output:
    wf_slices
    wf_start_utc
    '''
    raw_n = len(wf[0].data)
    pred_rate = int(pred_interval_sec/dt)
    full_len = int(pred_npts + pred_rate*\
        np.ceil(raw_n-pred_npts)/pred_rate)
    n_marching_win = int((full_len - pred_npts)/pred_rate)+1
    n_padded = full_len - raw_n

    wf = sac_len_complement(wf.copy(), max_length=full_len)
    pad_bef = pred_npts - pred_rate
    pad_aft = pred_npts
    for W in wf:
        W.data = np.insert(W.data, 0, np.zeros(pad_bef))
        W.data = np.insert(W.data, len(W.data), np.zeros(pad_aft))

    wf_n = []
    for w in range(3):
        wf_ = np.array([deepcopy(wf[w].data[pred_rate*i:pred_rate*i+pred_npts]) 
            for i in range(n_marching_win)])
        wf_dm = np.array([i-np.mean(i) for i in wf_])
        wf_std = np.array([np.std(i) for i in wf_dm])
        # reset std of 0 to 1 to prevent from ZeroDivisionError
        wf_std[wf_std==0] = 1
        wf_norm = np.array([wf_dm[i]/wf_std[i] for i in range(len(wf_dm))])
        wf_n.append(wf_norm)

    wf_slices = np.stack([wf_n[0], wf_n[1], wf_n[2]], -1)
    return np.array(wf_slices), pad_bef, pad_aft

def sac_len_complement(wf, max_length=None):
    '''Complement sac data into the same length
    '''
    wf_n = np.array([len(i.data) for i in wf])
    if not max_length:
        max_n = np.max(wf_n)
    else:
        max_n = max_length

    append_wf_id = np.where(wf_n!=max_n)[0]
    for w in append_wf_id:
        append_npts = max_n - len(wf[w].data)
        if append_npts > 0:
            for p in range(append_npts):
                wf[w].data = np.insert(
                    wf[w].data, len(wf[w].data), np.zeros(len(wf[w].data)))
        elif append_npts < 0:
            wf[w].data = wf[w].data[:max_n]
    return wf
          
def stream_standardize(st):
    '''
    input: obspy.stream object (raw data)
    output: obspy.stream object (standardized)
    '''
    st = st.detrend('demean')
    for s in st:
        s.data /= np.std(s.data)
    return st

def pred_MedianFilter(preds, masks, wf_npts, dt, 
        pred_npts, pred_interval_sec, pad_bef, pad_aft):
    ### 3. Integrate continuous predictions
    wf_n = wf_npts + (pad_bef+pad_aft)
    pred_array_P = [[] for _ in range(wf_n)]
    pred_array_S = [[] for _ in range(wf_n)]
    pred_array_mask = [[] for _ in range(wf_n)]
    pred_interval_pt =  int(round(pred_interval_sec/dt))

    init_pt = 0
    for i in range(len(preds)):
        pp = np.array_split(preds[i].T[0], pred_npts)
        ss = np.array_split(preds[i].T[1], pred_npts)
        mm = np.array_split(masks[i].T[0], pred_npts)
        j = 0
        for p, s, m in zip(pp, ss, mm):
            pred_array_P[init_pt+j].append(p)
            pred_array_S[init_pt+j].append(s)
            pred_array_mask[init_pt+j].append(m)
            j += 1
        init_pt += pred_interval_pt
    
    pred_array_P = np.array(pred_array_P, dtype='object')
    pred_array_S = np.array(pred_array_S, dtype='object')
    pred_array_mask = np.array(pred_array_mask, dtype='object')
    # fast revision of bottleneck
    lenP = np.array([len(p) for p in pred_array_P])
    nums = np.unique(lenP)
    array_P_med = np.zeros(wf_n)
    array_S_med = np.zeros(wf_n)
    array_M_med = np.zeros(wf_n)
    for k in nums:
        num_idx = np.where(lenP==k)[0]
        array_P_med[num_idx] = \
            np.median(np.hstack(
                np.take(pred_array_P, num_idx)), axis=0)
        array_S_med[num_idx] = \
            np.median(np.hstack(
                np.take(pred_array_S, num_idx)), axis=0)
        array_M_med[num_idx] = \
            np.median(np.hstack(
                np.take(pred_array_mask, num_idx)), axis=0)
    del pred_array_P
    del pred_array_S
    del pred_array_mask
    gc.collect()

    array_P_med = array_P_med[pad_bef:-pad_aft]
    array_S_med = array_S_med[pad_bef:-pad_aft]
    array_M_med = array_M_med[pad_bef:-pad_aft]
    assert len(array_P_med) == wf_npts

    return array_P_med, array_S_med, array_M_med

def pred_postprocess(array_P_med, array_S_med, array_M_med,
        dt=0.01, mask_trigger=[0.1, 0.1],  mask_len_thre=0.5,
        mask_err_win=0.5, trigger_thre=0.3):
    '''Predictions postprocessing
    1. filter the prediction functions using detection mask
    2. zero-padding non-detected space
    '''

    n_len = len(array_P_med) 
    mask_min_len = int(mask_len_thre/dt)
    err_win = int(mask_err_win/dt)
    p_evs = trigger_onset(
        array_M_med, mask_trigger[0], mask_trigger[1])
    # remove spikes
    p_evs = p_evs[(p_evs[:, 1]-p_evs[:, 0])>mask_min_len]
    # thresholded by mean detection value
    p_evs = p_evs[
        np.where( np.array([array_M_med[i[0]:i[1]].mean() 
        for i in p_evs]) > trigger_thre)[0] ]

    r_funM, r_funP, r_funS = \
        np.zeros(n_len), np.zeros(n_len), np.zeros(n_len)
    for i in p_evs:
        r_funM[i[0]:i[1]] +=\
            array_M_med[i[0]:i[1]]
        r_funP[i[0]-err_win:i[1]+err_win] += \
            array_P_med[i[0]-err_win:i[1]+err_win]
        r_funS[i[0]-err_win:i[1]+err_win] +=\
            array_S_med[i[0]-err_win:i[1]+err_win]
    return r_funP, r_funS, r_funM

class PhasePicker:
    def __init__(
        self,
        model=None,
        dt=0.01,
        pred_npts=2001, 
        pred_interval_sec=4,
        postprocess_config={
            'mask_trigger': [0.1, 0.1], 
            'mask_len_thre': 0.5,
            'mask_err_win':0.5, 
            'trigger_thre':0.3
            }
        ):

        self.model = model
        self.dt = dt
        self.pred_npts = pred_npts
        self.pred_interval_sec = pred_interval_sec
        self.postprocess_config = postprocess_config

        if model == None:
            AssertionError("The Phase picker model should be defined!")

    def predict(self, wf=None, postprocess=False):
        from time import time
        if wf==None:
            AssertionError("Obspy.stream should be assigned as `wf=?`!")
        if len(wf[0].data) < self.pred_npts:
            AssertionError(
                f"Data should be longer than {self.pred_npts} points.")
        ## store continuous data into array according to prediction interval
        wf_slices, pad_bef, pad_aft = conti_standard_wf_fast(wf, 
            pred_npts=self.pred_npts, 
            pred_interval_sec=self.pred_interval_sec, dt=self.dt)
        ## make prediction
        t1 = time()
        predPhase, masks = self.model.predict(wf_slices)
        del wf_slices
        gc.collect()
        #print(f"Prediction making: {time()-t1:.2f} secs")
        ## apply median filter to sliding predictions
        wf_npts = len(wf[0].data)
        array_P_med, array_S_med, array_M_med = pred_MedianFilter(
           preds=predPhase, masks=masks, wf_npts=wf_npts, 
           dt=self.dt, pred_npts=self.pred_npts, 
           pred_interval_sec=self.pred_interval_sec,
           pad_bef=pad_bef, pad_aft=pad_aft)

        del predPhase
        del masks
        gc.collect()           
        # replace nan by 0
        find_nan = np.where(np.isnan(array_M_med))[0]
        array_P_med[find_nan] = np.zeros(len(find_nan))
        array_S_med[find_nan] = np.zeros(len(find_nan))
        array_M_med[find_nan] = np.zeros(len(find_nan))
        # replace inf by 0
        find_inf = np.where(np.isnan(array_M_med))[0]
        array_P_med[find_inf] = np.zeros(len(find_inf))
        array_S_med[find_inf] = np.zeros(len(find_inf))
        array_M_med[find_inf] = np.zeros(len(find_inf))
         
        if postprocess:
            array_P_med, array_S_med, array_M_med = pred_postprocess(
                array_P_med, array_S_med, array_M_med, dt=self.dt,
                **self.postprocess_config)
                
        return array_P_med, array_S_med, array_M_med
            
if __name__ == '__main__':
    pass
