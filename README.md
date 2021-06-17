# recording-script-generator

File structure:

```txt
corpora
├── ljs
│  ├── original
|  │  ├── data.json
│  ├── normalized
|  │  ├── data.json
│  ├── normalized+normalized_ipa
|  │  ├── data.json
│  ├── ...
├── ...
scripts
├── ljs_ipa
|  ├── data.json
│  ├── default_script
|  │  ├── selection.json
|  │  ├── selected.csv
|  │  ├── ignored.csv
|  │  └── rest.csv
│  ├── final_script
|  │  ├── selection.json
|  │  ├── selected.csv
|  │  ├── ignored.csv
|  │  └── rest.csv
│  ├── ...
├── ...
```

Example content of corpus `corpus.txt`:

```txt
ab
cd
ef
```

Example content of corpora `data.json`:

```json
{
  "reading_passages_lang": 0,
  "reading_passages": {
    "0": ["a", "b"],
    "1": ["c", "d"],
    "2": ["e", "f"],
  },
  "representation_lang": 0,
  "representation": {
    "0": ["a", "b"],
    "1": ["c", "d"],
    "2": ["e", "f"],
  },
}
```

Example content of script `data.json`:

```json
{
  "reading_passages": {
    "0": ["a", "b"],
    "1": ["c", "d"],
    "2": ["e", "f"],
  },
  "representation": {
    "0": ["a", "b"],
    "1": ["c", "d"],
    "2": ["e", "f"],
  },
}
```

Example content of script `selection.json`:

```json
{
  "selected": [0],
  "ignored": [1],
  "rest": [2],
}
```

Example content of `selected.csv`:

```csv
2\t"001"\t"ab"
4\t"002"\t"cd"
94\t"003"\t"ef"
```

Each line contains one utterance.

**Methods**

- **ljs_to_text**(corpus_name, step_name, out_path)
- **add_corpus_from_text**(corpus_name, text_path, lang)
- **normalize**(corpus_name, in_step_name, out_step_name, mode=reading_passage/representation/both, overwrite)
- **convert_to_ipa**(corpus_name, in_step_name, out_step_name, mode=reading_passage/representation/both, options=IPAOptions, overwrite)
- **print_stats**(corpus_name, step_name, estimated_symbols_per_s)
  - prints u.a. the potential duration of the script

**Merge methods**

- **merge**(corpora, merge_name, script_name):
  - corpora: `"{corpus_name},{step_name};..."` -> selects the selected utterances
  - all corpora do not need to have the same lang for reading and representation
- **select_rest**(merge_name, in_script_name, out_script_name)
- **select_greedy_ngrams_epochs**(merge_name, in_script_name, out_script_name, ngram, epochs, ignore_symbols)
  - selects from those ids that are not `selected` or `ignored`.
- **select_greedy_ngrams_minutes**(merge_name, in_script_name, out_script_name, ngram, minutes, ignore_symbols, estimated_symbols_per_s)
- **filter**(merge_name, in_script_name, out_script_name, min_len, max_len, not_allowed_symbols)
  - adds the utterances which have less chars or more chars to ignored
- **print_stats**(merge_name, script_name, estimated_symbols_per_s)
  - prints u.a. the potential duration of the script
  - prints all symbols
- **merge_merged**(in_merge_names, out_merge_name)
  - merges merged corpora
  - in_merge_names: `"{merge_name};..."`

