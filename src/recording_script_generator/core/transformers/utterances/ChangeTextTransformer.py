from recording_script_generator.core.types import Utterances
from text_utils import change_symbols
from tqdm import tqdm


class ChangeTextTransformer():
  def fit(self, remove_space_around_punctuation: bool, n_jobs: int, chunksize: int):
    self.remove_space_around_punctuation = remove_space_around_punctuation
    self.n_jobs = n_jobs = n_jobs = n_jobs
    self.chunksize = chunksize

  def transform(self, utterances: Utterances) -> Utterances:
    res = utterances.copy()
    for utterance_id, symbols in tqdm(utterances.items()):
      new_symbols = change_symbols(
        symbols=symbols,
        remove_space_around_punctuation=self.remove_space_around_punctuation,
        lang=utterances.language,
      )

      res[utterance_id] = new_symbols

    return res
