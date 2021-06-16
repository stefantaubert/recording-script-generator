from pathlib import Path

from recording_script_generator.app.preparation import (
    add_corpus_from_text_file, app_convert_to_ipa, app_normalize)
from recording_script_generator.core.preparation import PreparationTarget
from text_utils.ipa2symb import IPAExtractionSettings
from text_utils.language import Language
from text_utils.text import EngToIpaMode

if __name__ == "__main__":
  base_dir = Path("/home/mi/data/recording-script-generator")

  add_corpus_from_text_file(
    base_dir=base_dir,
    corpus_name="ljs_debug",
    ipa_settings=None,
    lang=Language.ENG,
    step_name="initial",
    text_path=Path("/tmp/ljs.txt"),
    overwrite=True,
  )

  app_normalize(
    base_dir=base_dir,
    corpus_name="ljs_debug",
    in_step_name="initial",
    out_step_name="en_norm",
    ipa_settings=None,
    target=PreparationTarget.BOTH,
  )

  app_convert_to_ipa(
    base_dir=base_dir,
    corpus_name="ljs_debug",
    in_step_name="initial",
    out_step_name="en_norm+ipa_norm",
    ipa_settings=IPAExtractionSettings(True, True, "_"),
    target=PreparationTarget.REPRESENTATIONS,
    mode=EngToIpaMode.BOTH,
    replace_unknown_with="_",
    use_cache=True,
  )
