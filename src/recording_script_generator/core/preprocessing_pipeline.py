from logging import getLogger
from math import inf
from multiprocessing.pool import RemoteTraceback
from sys import getsizeof
from typing import Dict, cast

from recording_script_generator.core.estimators.selection import *
from recording_script_generator.core.estimators.utterances import *
from recording_script_generator.core.transformers.selection.AddTransformer import \
    AddTransformer
from recording_script_generator.core.transformers.selection.SelectTransformer import \
    SelectTransformer
from recording_script_generator.core.transformers.utterances import (
    ArpaToIpaTransformer, ChangeIpaTransformer, ChangeTextTransformer,
    EngToArpaTransformer, NormalizeTransformer, RemoveTransformer)
from recording_script_generator.core.types import (ReadingPassages,
                                                   Representations, Selection,
                                                   Utterance, UtteranceId,
                                                   Utterances)


def handle_error(exception: Exception) -> None:
  logger = getLogger(__name__)
  #tb = sys.exc_info()
  # traceback.print_stack()
  # print(traceback.format_exc())
  logger.exception(exception)
  remote_traceback = cast(RemoteTraceback, exception.__cause__)
  logger.info(remote_traceback.tb)
  pass


def do_pipeline_prepare(reading_passages: ReadingPassages, representations: Representations, selection: Selection, n_jobs: int, chunksize: int, maxtasksperchild: int):
  logger = getLogger(__name__)

  step = AcronymEstimator()
  step.fit(n_jobs, maxtasksperchild, chunksize)
  remove = step.estimate(representations)

  step = RemoveTransformer()
  step.fit()
  representations = step.transform(representations, remove)

  step = NormalizeTransformer()
  step.fit()
  representations = step.transform(representations)

  step = DuplicateEstimator()
  step.fit()
  remove = step.estimate(representations)

  step = RemoveTransformer()
  step.fit()
  representations = step.transform(representations, remove)

  step = UnfrequentWordCountEstimator()
  step.fit(2, n_jobs, maxtasksperchild, chunksize)
  remove = step.estimate(representations)

  step = RemoveTransformer()
  step.fit()
  representations = step.transform(representations, remove)

  step = ProperNameEstimator()
  step.fit(n_jobs, maxtasksperchild, chunksize)
  remove = step.estimate(representations)

  step = RemoveTransformer()
  step.fit()
  representations = step.transform(representations, remove)

  step = UndesiredTextEstimator()
  undesired = set("/ \\ - : @ ; * % \" ( ) [ ] { } quote oswald ye hath pp.".split(" "))
  step.fit(undesired, n_jobs, maxtasksperchild, chunksize)
  remove = step.estimate(representations)

  step = RemoveTransformer()
  step.fit()
  representations = step.transform(representations, remove)

  step = WordCountEstimator()
  step.fit(3, None, n_jobs, maxtasksperchild, chunksize)
  remove = step.estimate(representations)

  step = RemoveTransformer()
  step.fit()
  representations = step.transform(representations, remove)

  step = UnknownWordEstimator()
  step.fit(0, n_jobs, maxtasksperchild, chunksize)
  remove = step.estimate(representations)

  step = RemoveTransformer()
  step.fit()
  representations = step.transform(representations, remove)

  step = EngToArpaTransformer()
  step.fit(representations, n_jobs, chunksize)
  representations = step.transform(representations)

  step = UndesiredTextEstimator()
  undesired = {"'"}
  step.fit(undesired, n_jobs, maxtasksperchild, chunksize)
  remove = step.estimate(representations)

  step = RemoveTransformer()
  step.fit()
  representations = step.transform(representations, remove)

  step = ArpaToIpaTransformer()
  step.fit(n_jobs, chunksize)
  representations = step.transform(representations)

  # Sync reading passages
  step = RemoveTransformer()
  step.fit()
  remove = reading_passages.keys() - representations.keys()
  reading_passages = step.transform(reading_passages, remove)

  return reading_passages, representations, selection


def do_pipeline_select(reading_passages: ReadingPassages, representations: Representations, selection: Selection, n_jobs: int, chunksize: int, maxtasksperchild: int):
  logger = getLogger(__name__)

  logger.info(f"Size of utterances in memory: {getsizeof(representations)/1024**3:.2f} Gb")

  utterance_durations_s = get_utterance_durations_based_on_symbols(
    reading_passages, reading_speed_symbols_per_s=14)

  # get selected
  step = DeselectedEstimator()
  step.fit()
  deselected = step.estimate(selection, representations)

  step = RemoveTransformer()
  step.fit()
  deselected_utterances = step.transform(representations, selection)
  selected_utterances = step.transform(representations, deselected)

  step = KldDurationEstimator()
  step.fit(n_gram=1, minutes=10, ignore_symbols=" ", boundary=(8, inf))
  add = step.estimate(deselected_utterances, selected_utterances, utterance_durations_s)

  step = AddTransformer()
  step.fit()
  selection = step.transform(selection, add, representations)

  step = DeselectedEstimator()
  step.fit()
  deselected = step.estimate(selection, representations)

  step = RemoveTransformer()
  step.fit()
  deselected_utterances = step.transform(representations, selection)
  selected_utterances = step.transform(representations, deselected)

  step = KldDurationEstimator()
  step.fit(n_gram=1, minutes=13, ignore_symbols=" ", boundary=(4, 8))
  add = step.estimate(deselected_utterances, selected_utterances, utterance_durations_s)

  step = AddTransformer()
  step.fit()
  selection = step.transform(selection, add, representations)

  step = DeselectedEstimator()
  step.fit()
  deselected = step.estimate(selection, representations)

  step = RemoveTransformer()
  step.fit()
  deselected_utterances = step.transform(representations, selection)
  selected_utterances = step.transform(representations, deselected)

  step = KldDurationEstimator()
  step.fit(n_gram=1, minutes=10, ignore_symbols=" ", boundary=(0, 4))
  add = step.estimate(deselected_utterances, selected_utterances, utterance_durations_s)

  step = AddTransformer()
  step.fit()
  selection = step.transform(selection, add, representations)

  # Remove non-selected utterances

  step = DeselectedEstimator()
  step.fit()
  deselected = step.estimate(selection, representations)

  step = RemoveTransformer()
  step.fit()
  representations = step.transform(representations, deselected)

  # Sync reading passages

  remove = reading_passages.keys() - representations.keys()
  reading_passages = step.transform(reading_passages, remove)

  return reading_passages, representations, selection


def get_utterance_durations_based_on_symbols(utterances: Utterances, reading_speed_symbols_per_s: float) -> Dict[UtteranceId, float]:
  durations = {
    utterance_id: len(symbols) / reading_speed_symbols_per_s
    for utterance_id, symbols in utterances.items()
  }
  return durations
