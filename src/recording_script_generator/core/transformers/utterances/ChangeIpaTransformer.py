from recording_script_generator.core.types import Utterances, clone_utterances
from text_utils import change_ipa as change_ipa_method
from tqdm import tqdm


class ChangeIpaTransformer():
  def fit(self, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, break_n_thongs: bool, build_n_thongs: bool, n_jobs: int, chunksize: int):
    self.ignore_tones = ignore_tones
    self.ignore_arcs = ignore_arcs
    self.ignore_stress = ignore_stress
    self.break_n_thongs = break_n_thongs
    self.build_n_thongs = build_n_thongs
    self.n_jobs = n_jobs = n_jobs = n_jobs
    self.chunksize = chunksize

  def transform(self, utterances: Utterances) -> Utterances:
    res = clone_utterances(utterances)
    
    for utterance_id, symbols in tqdm(utterances.items()):
      new_symbols = change_ipa_method(
        symbols=symbols,
        ignore_tones=self.ignore_tones,
        ignore_arcs=self.ignore_arcs,
        ignore_stress=self.ignore_stress,
        break_n_thongs=self.break_n_thongs,
        build_n_thongs=self.build_n_thongs,
        language=utterances.language,
      )

      res[utterance_id] = new_symbols

    return res
