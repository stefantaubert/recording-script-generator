# Remote Dependencies

- text-utils
  - pronunciation_dict_parser
  - g2p_en
  - sentence2pronunciation
- text-selection

## Pipfile

### Local

```Pipfile
text-utils = {editable = true, path = "./../text-utils"}
text-selection = {editable = true, path = "./../text-selection"}

pronunciation_dict_parser = {editable = true, path = "./../pronunciation_dict_parser"}
g2p_en = {editable = true, path = "./../g2p"}
sentence2pronunciation = {editable = true, path = "./../sentence2pronunciation"}
```

### Remote

```Pipfile
text-utils = {editable = true, ref = "master", git = "https://github.com/stefantaubert/text-utils.git"}
text-selection = {editable = true, ref = "master", git = "https://github.com/stefantaubert/text-selection.git"}
```

## setup.cfg

```cfg
text_utils@git+https://github.com/stefantaubert/text-utils.git@master
text_selection@git+https://github.com/stefantaubert/text-selection.git@master
```
