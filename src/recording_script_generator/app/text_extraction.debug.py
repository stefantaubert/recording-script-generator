from pathlib import Path

from recording_script_generator.app.text_extraction import app_ljs_to_text

if __name__ == "__main__":
  app_ljs_to_text(
    ljs_path=Path("/data/datasets/LJSpeech-1.1"),
    output_file=Path("/tmp/out.txt"),
  )
