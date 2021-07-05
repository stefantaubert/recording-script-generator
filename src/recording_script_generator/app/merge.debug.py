from pathlib import Path

from recording_script_generator.app.merge import (
    app_log_stats, app_merge, app_select_greedy_ngrams_epochs, app_select_rest)
from recording_script_generator.app.preparation import (
    add_corpus_from_text_file, app_convert_to_ipa, app_normalize)
from recording_script_generator.core.preparation import PreparationTarget
from text_utils.ipa2symb import IPAExtractionSettings
from text_utils.language import Language
from text_utils.text import EngToIpaMode

if __name__ == "__main__":
  base_dir = Path("out")

  app_merge(
    base_dir=base_dir,
    corpora=[("ljs_debug", "en_norm")],
    merge_name="ljs_debug",
    script_name="initial",
    overwrite=True,
  )

  app_select_greedy_ngrams_epochs(
    base_dir=base_dir,
    merge_name="ljs_debug",
    in_script_name="initial",
    out_script_name="all",
    epochs=5,
    n_gram=1,
    overwrite=False,
  )

  app_log_stats(
    base_dir=base_dir,
    merge_name="ljs_debug",
    script_name="all",
  )

  # app_select_rest(
  #   base_dir=base_dir,
  #   merge_name="ljs_debug",
  #   in_script_name="initial",
  #   out_script_name="all",
  # )
