# ARRU_seismic_backprojection

This repo is the official implementation of "Towards fully autonomous seismic networks: backprojecting deep-learning-based phase time functions for earthquake monitoring on continuous recordings".<br/>

In this repo we provide template codes that backprojects seismic phase-time functions with pre-calculated [travel-time tables](https://drive.google.com/file/d/1OADPD0nwAeX5W843Wt9E6I5K8MiYS7nM/view?usp=sharing). The outputs of following scripts could be retreived [here](https://drive.google.com/file/d/101h8nZopPDV86DnYMxZEwJ7nj293Q1Z-/view?usp=sharing). <br/>

```$tar -zvxf out_data.tar.gz$``` <br/>
```$tar -zvxf metadata.tar.gz$``` <br/>

**Do seismic phase picking on 1-hour-long seismograms using [ARRU phase picker](https://github.com/tso1257771/Attention-Recurrent-Residual-U-Net-for-earthquake-detection)**<br/>
This script generates phase-time functions of raw seismograms in SAC format.<br/>
```$ python P01_continuous_pred.py```<br/>

**Convert phase-time functions into binary**<br/>
```$ python P02_ARRU_sac2bin.py```<br/>

**Do seismic backprojection using prepared travel-time tables and phase-time fuinctions**<br/>
```$ python P03_ARRU_BP_PSwin_MPI.py```<br/>

**Find potential earthquake events**<br/>
```$ python P04_find_potential_events.py```

# Reference
Wu‐Yu Liao, En‐Jui Lee, Dawei Mu, Po Chen, Ruey‐Juin Rau; ARRU Phase Picker: Attention Recurrent‐Residual U‐Net for Picking Seismic P‐ and S‐Phase Arrivals. Seismological Research Letters 2021; doi: https://doi.org/10.1785/0220200382
