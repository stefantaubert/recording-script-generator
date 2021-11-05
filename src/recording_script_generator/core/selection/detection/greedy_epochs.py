from typing import Optional, Set

from ordered_set import OrderedSet
from recording_script_generator.core.types import UtteranceId, Utterances
from text_selection import greedy_ngrams_epochs
from text_utils.types import Symbol


def get_utterances_through_greedy_epochs(utterances: Utterances, n_gram: int, epochs: int, ignore_symbols: Optional[Set[Symbol]]) -> OrderedSet[UtteranceId]:
  new_selected = greedy_ngrams_epochs(
    data=utterances,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    epochs=epochs,
  )
  return new_selected
