import json
import os
import pickle
from pathlib import Path
from typing import Any, List, Optional, Tuple


def get_subdir(training_dir_path: str, subdir: str, create: bool = True) -> str:
  result = os.path.join(training_dir_path, subdir)
  if create:
    os.makedirs(result, exist_ok=True)
  return result


def read_lines(path: str) -> List[str]:
  assert os.path.isfile(path)
  with open(path, "r", encoding='utf-8') as f:
    lines = f.readlines()
  res = [x.strip("\n") for x in lines]
  return res


def read_text(path: Path) -> str:
  assert path.is_file()
  with path.open("r", encoding='utf-8') as f:
    text = f.read()
  return text


def parse_json(path: str) -> Any:
  assert os.path.isfile(path)
  with open(path, 'r', encoding='utf-8') as f:
    tmp = json.load(f)
  return tmp


def save_json(path: str, mapping_dict: Any) -> None:
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(mapping_dict, f, ensure_ascii=False, indent=2)


def save_obj(path: Path, obj: Any) -> None:
  assert path.parent.exists()
  with path.open(mode="wb") as f:
    pickle.dump(obj=obj, file=f)


def load_obj(path: Path) -> Any:
  with path.open(mode="rb") as f:
    obj = pickle.load(file=f)
  return obj


def save_lines(path: Path, lines: List[str]) -> None:
  assert path.parent.exists()
  path.write_text("\n".join(lines), encoding='utf-8')


def parse_tuple_list(tuple_list: Optional[str] = None) -> Optional[List[Tuple]]:
  """ tuple_list: "a,b;c,d;... """
  if tuple_list is None:
    return None

  step1: List[str] = tuple_list.split(';')
  result: List[Tuple] = [tuple(x.split(',')) for x in step1]
  result = list(sorted(set(result)))
  return result
