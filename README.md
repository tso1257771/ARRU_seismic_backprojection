# ARRU_seismic_backprojection

This repo is the official implementation of "Towards fully autonomous seismic networks: backprojecting deep-learning-based phase time functions for earthquake monitoring on continuous recordings".<br/>

In this repo we provide template codes that backprojects seismic phase-time functions with pre-calculated [travel-time tables](https://drive.google.com/file/d/1OADPD0nwAeX5W843Wt9E6I5K8MiYS7nM/view?usp=sharing). The outputs of following scripts could be retreived [here](https://drive.google.com/file/d/101h8nZopPDV86DnYMxZEwJ7nj293Q1Z-/view?usp=sharing). Download and uncompressed them. <br/>
```$tar -zvxf out_data.tar.gz``` <br/>
```$tar -zvxf metadata.tar.gz``` <br/>

**Step 1. Do seismic phase picking on 1-hour-long seismograms using [ARRU phase picker](https://github.com/tso1257771/Attention-Recurrent-Residual-U-Net-for-earthquake-detection)**<br/>
This script generates phase-time functions of raw seismograms in SAC format.<br/>
```$ python P01_continuous_pred.py```<br/>

**Step 2. Convert phase-time functions into binary**<br/>
```$ python P02_ARRU_sac2bin.py```<br/>

**Step 3. Do seismic backprojection using prepared travel-time tables and phase-time functions**<br/>
```$ python P03_ARRU_BP_PSwin_MPI.py```<br/>

**Step 4. Find potential earthquake events**<br/>
```$ python P04_find_potential_events.py```

While the postprocess of backprojection results are tedious. in this repo we only provide main scripts of seismic phase picking and backprojection. 
The full catalog of our work for July, 2019 Ridgecrest earthquake sequence is available at: ```./ARRU_BP_201907_catalog_final.txt```<br/>

# Reference
Wu‐Yu Liao, En‐Jui Lee, Dawei Mu, Po Chen, Ruey‐Juin Rau; ARRU Phase Picker: Attention Recurrent‐Residual U‐Net for Picking Seismic P‐ and S‐Phase Arrivals. Seismological Research Letters 2021; doi: https://doi.org/10.1785/0220200382

Wu‐Yu Liao, En‐Jui Lee, Dawei Mu, Po Chen; Toward Fully Autonomous Seismic Networks: Backprojecting Deep Learning‐Based Phase Time Functions for Earthquake Monitoring on Continuous Recordings. Seismological Research Letters 2022; doi: https://doi.org/10.1785/0220210274
