import random
import string
from multiprocessing import Pool
from typing import Tuple

from recording_script_generator.core.types import Utterances
from recording_script_generator.core.utterances.inspection.undesired_text import (
    get_utterances_with_undesired_text, main)
from tqdm import tqdm


def test_containing_text__returns_true():
  result = main(
    symbols=("When you took off that night eight weeks ago, that night I kissed you good-by, your ship ... oh don't you comprehend?..."),
    undesired={"..."}
  )

  assert result


def random_string_generator(str_size: int, allowed_chars: str):
  return ''.join(random.choice(allowed_chars) for x in range(str_size))


chars = string.ascii_letters  # + string.punctuation


def get_random_sentence(words_count: int) -> str:
  words = []
  for _ in range(words_count):
    words.append(random_string_generator(random.randint(3, 10), chars))
  return ' '.join(words)


def mp_get_random_sentence(i: int) -> Tuple[int, str]:
  return i, get_random_sentence(random.randint(3, 20))


def test_get_utterances_with_undesired_text():
  utterances = Utterances({
    1: tuple("this is a test."),
    2: tuple("this is a second test."),
  })

  result = get_utterances_with_undesired_text(
    utterances=utterances,
    undesired={"second"},
    chunksize=1,
    maxtasksperchild=1,
    n_jobs=1,
  )

  assert result == {2}


def test_get_utterances_with_undesired_text__stress_test():
  n_jobs = 16
  maxtasksperchild = None
  chunksize = 10000
  utterances = 1000000
  random.seed(1234)

  print("Generating test sentences...")
  with Pool(processes=n_jobs, maxtasksperchild=maxtasksperchild) as pool:
    utterances = Utterances(pool.map(mp_get_random_sentence,
                            range(utterances), chunksize=chunksize))
  print("Done generating")

  result = get_utterances_with_undesired_text(
    utterances=utterances,
    undesired={"a"},
    chunksize=chunksize,
    maxtasksperchild=maxtasksperchild,
    n_jobs=n_jobs,
  )

  assert len(result) == utterances
