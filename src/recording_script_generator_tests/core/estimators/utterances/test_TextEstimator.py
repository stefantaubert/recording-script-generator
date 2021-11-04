
import random
import string
from multiprocessing import Pool
from typing import Tuple

from recording_script_generator.core.estimators.utterances.TextEstimator import \
    TextEstimator
from recording_script_generator.core.types import Utterance, Utterances
from tqdm import tqdm


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


def test_estimate():
  instance = TextEstimator()
  instance.fit(
    undesired={"second"},
    chunksize=1,
    maxtasksperchild=1,
    n_jobs=1,
  )

  utterances = Utterances({
    1: tuple("this is a test."),
    2: tuple("this is a second test."),
  })

  result = instance.estimate(utterances)

  assert result == {2}


def test_estimate__stress_test():
  n_jobs = 16
  chunksize = 10000
  utterances = 1000000
  instance = TextEstimator()
  instance.fit(
    undesired={"a"},
    chunksize=chunksize,
    maxtasksperchild=1,
    n_jobs=n_jobs,
  )

  random.seed(1234)
  with Pool(processes=n_jobs, maxtasksperchild=1) as pool:
    utterances = Utterances(pool.map(mp_get_random_sentence,
                            range(utterances), chunksize=chunksize))
  print("Done generating")

  result = instance.estimate(utterances)

  # assert result == {2}
