# recording-script-generator

## Installation

```sh
pipenv install --skip-lock
pipenv run python -m nltk.downloader punkt
# pipenv run python -m nltk.downloader brown
# pipenv run python -m nltk.downloader wordnet
```

## Example usage

```sh
export base_dir="../out"
export PYTHONPATH="./src"
cd $PYTHONPATH

pipenv run python -m cli add-file \
  --corpus_name="ljs" \
  --step_name="init" \
  --text_path="../corpora/ljs.txt" \
  --lang=ENG \
  --overwrite

pipenv run python -m cli add-file \
  --corpus_name="darwin" \
  --step_name="init" \
  --text_path="../corpora/darwin.txt" \
  --lang=ENG \
  --overwrite

pipenv run python -m cli merge \
  --corpora_step_names="ljs,init;darwin,init" \
  --out_corpus_name="merged" \
  --out_step_name="init" \
  --overwrite

export corpus_name="merged"
export apply_step_name="preprocessed"

pipenv run python -m cli remove-acronyms \
  --corpus_name="$corpus_name" \
  --in_step_name="init" \
  --target=READING_PASSAGES \
  --out_step_name="$apply_step_name" \
  --overwrite

pipenv run python -m cli normalize \
  --corpus_name="$corpus_name" \
  --in_step_name="$apply_step_name" \
  --target=BOTH \
  --overwrite

pipenv run python -m cli remove-rare-words \
  --corpus_name="$corpus_name" \
  --in_step_name="$apply_step_name" \
  --target=READING_PASSAGES \
  --min_occurrence_count=2 \
  --overwrite

pipenv run python -m cli remove-proper-names \
  --corpus_name="$corpus_name" \
  --in_step_name="$apply_step_name" \
  --target=READING_PASSAGES \
  --overwrite

pipenv run python -m cli remove-text \
  --corpus_name="$corpus_name" \
  --in_step_name="$apply_step_name" \
  --target=READING_PASSAGES \
  --undesired="/ \\ - : @ ; * % \" ( ) [ ] { } quote oswald" \
  --overwrite

pipenv run python -m cli remove-duplicates \
  --corpus_name="$corpus_name" \
  --in_step_name="$apply_step_name" \
  --target=READING_PASSAGES \
  --overwrite

pipenv run python -m cli remove-word-counts \
  --corpus_name="$corpus_name" \
  --in_step_name="$apply_step_name" \
  --target=READING_PASSAGES \
  --min_word_count=3 \
  --overwrite

pipenv run python -m cli remove-unknown-words \
  --corpus_name="$corpus_name" \
  --in_step_name="$apply_step_name" \
  --target=READING_PASSAGES \
  --max_unknown_word_count=0 \
  --overwrite

pipenv run python -m cli to-ipa \
  --corpus_name="$corpus_name" \
  --in_step_name="$apply_step_name" \
  --out_step_name="ipa" \
  --target=REPRESENTATIONS \
  --mode=BOTH \
  --overwrite

pipenv run python -m cli remove-text \
  --corpus_name="$corpus_name" \
  --in_step_name="ipa" \
  --undesired="'" \
  --target=REPRESENTATIONS \
  --overwrite

export corpus_name="merged"
export apply_step_name="selection"

pipenv run python -m cli select-kld-duration \
  --corpus_name="$corpus_name" \
  --in_step_name="ipa" \
  --out_step_name="$apply_step_name" \
  --n_gram=1 \
  --minutes=3 \
  --ignore_symbols=" &'" \
  --boundary_min_s=0 \
  --boundary_max_s=3 \
  --overwrite

pipenv run python -m cli select-kld-duration \
  --corpus_name="$corpus_name" \
  --in_step_name="$apply_step_name" \
  --n_gram=1 \
  --minutes=12 \
  --ignore_symbols=" &'" \
  --boundary_min_s=3 \
  --boundary_max_s=5 \
  --overwrite

pipenv run python -m cli select-kld-duration \
  --corpus_name="$corpus_name" \
  --in_step_name="$apply_step_name" \
  --n_gram=1 \
  --minutes=15 \
  --ignore_symbols=" &'" \
  --boundary_min_s=5 \
  --overwrite

```

## File structure

```txt
$base_dir
├── ljs
│  ├── original
|  │  ├── data.pkl
|  │  ├── selected.csv
|  │  ├── selected.txt
|  │  ├── selected.tex
|  │  ├── rest.csv
|  │  ├── rest.txt
|  │  ├── rest.tex
|  │  ├── 1_gram_stats.csv
|  │  ├── 2_gram_stats.csv
|  │  ├── 3_gram_stats.csv
│  ├── ...
├── ...
```

Example content of `selected.txt`:

```txt
001: There was no longer the faintest possible excuse for overcrowding.
002: Anniversary, platform, and lyceum lectures have much in common.
003: Wherefore they were happy, and rejoiced.
004: My muse, in sooth!
005: God be thanked therefore!
006: The soldiers then?
007: Five tablespoonfuls of yeast.
008: The watch, ho!
009: Burch's trunk was there.
```

Each line contains one utterance.

## Template corpora

- LJSpeech data
- Charles Darwin: On the Origin of Species, 6th Edition (1872)
- ...
