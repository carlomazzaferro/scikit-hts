import os
from multiprocessing import cpu_count

n_cores = int(os.getenv("NUMBER_OF_CPUS") or cpu_count())

MODEL = 'prophet'
REVISION = 'OLS'
LOW_MEMORY = False
CHUNKSIZE = None
N_PROCESSES = max(1, n_cores // 2)
PROFILING = False
DISABLE_PROGRESSBAR = False
SHOW_WARNINGS = False
PARALLELISATION = True
