# ARRU_seismic_backprojection

**Do seismic phase picking on 1-hour-long seismograms**<br/>
>Input/Output: SAC files <br/>

```$ python P01_continuous_pred.py```<br/>

**Convert phase-time functions into binary**<br/>
```$ python P02_ARRU_sac2bin.py```<br/>

**Do seismic backprojection using prepared travel-time tables and phase-time fuinctions**<br/>
```$ python P03_ARRU_BP_PSwin_MPI.py```<br/>

**Find potential earthquake events**<br/>
```$ python P04_find_potential_events.py```

# Reference
Wu‐Yu Liao, En‐Jui Lee, Dawei Mu, Po Chen, Ruey‐Juin Rau; ARRU Phase Picker: Attention Recurrent‐Residual U‐Net for Picking Seismic P‐ and S‐Phase Arrivals. Seismological Research Letters 2021; doi: https://doi.org/10.1785/0220200382
