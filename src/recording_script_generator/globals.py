from math import inf
from multiprocessing import cpu_count

from recording_script_generator.core.exporting import SortingMode

DEFAULT_SEED = 1111
DEFAULT_SORTING_MODE = SortingMode.BY_SELECTION
DEFAULT_AVG_CHARS_PER_S = 13.7
DEFAULT_IGNORE = {}
DEFAULT_SPLIT_BOUNDARY_MIN_S = 0
DEFAULT_SPLIT_BOUNDARY_MAX_S = inf
SEP = "\t"
DEFAULT_N_JOBS = cpu_count()
DEFAULT_CHUNKSIZE_FILES = 100
DEFAULT_CHUNKSIZE_UTTERANCES = 1000000
DEFAULT_MAXTASKSPERCHILD = None
