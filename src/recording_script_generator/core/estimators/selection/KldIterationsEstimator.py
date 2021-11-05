from typing import Optional, Set

from ordered_set import OrderedSet
from recording_script_generator.core.types import UtteranceId, Utterances
from text_selection import greedy_kld_uniform_ngrams_iterations
from text_utils.types import Symbol


def get_utterances_through_kld_iterations(utterances: Utterances, n_gram: int, iterations: int, ignore_symbols: Optional[Set[Symbol]]) -> OrderedSet[UtteranceId]:
  new_selected = greedy_kld_uniform_ngrams_iterations(
    data=utterances,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    iterations=iterations,
  )
  return new_selected


class KldIterationsEstimator():
  def fit(self, n_gram: int, iterations: int, ignore_symbols: Optional[Set[Symbol]]):
    self.n_gram = n_gram
    self.iterations = iterations
    self.ignore_symbols = ignore_symbols

  def estimate(self, utterances: Utterances) -> OrderedSet[UtteranceId]:
    new_selected = greedy_kld_uniform_ngrams_iterations(
      data=utterances,
      n_gram=self.n_gram,
      ignore_symbols=self.ignore_symbols,
      iterations=self.iterations,
    )

    return new_selected
