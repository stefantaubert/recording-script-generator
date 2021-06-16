from pathlib import Path

from recording_script_generator.app.text_extraction import app_ljs_to_text

if __name__ == "__main__":
  text_out = Path("/tmp/ljs.txt")
  app_ljs_to_text(
    ljs_path=Path("/data/datasets/LJSpeech-1.1"),
    output_file=text_out,
  )

  res = text_out.read_text()
  text_out.write_text(res[:10000])
