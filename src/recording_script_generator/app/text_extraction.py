from logging import getLogger
from pathlib import Path

from recording_script_generator.core.text_extraction import ljs_to_text
from recording_script_generator.utils import save_lines
from speech_dataset_parser import parse_ljs


def app_ljs_to_text(ljs_path: Path, output_file: Path):
  logger = getLogger(__name__)
  data = parse_ljs(
    dir_path=ljs_path,
    logger=logger,
  )
  res = ljs_to_text(data)
  save_lines(output_file, res)
  logger.info("Done.")
