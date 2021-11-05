from ordered_set import OrderedSet
from recording_script_generator.core.transformers.selection.AddTransformer import (
    AddTransformer, get_selection_added)
from recording_script_generator.core.transformers.selection.RemoveTransformer import (
    RemoveTransformer, get_selection_removed)
from recording_script_generator.core.types import (Selection, UtteranceId,
                                                   Utterances)


def get_selection_set(selection: Selection, select: OrderedSet[UtteranceId], utterances: Utterances) -> Selection:
  remove = selection - select
  add = select - selection
  selection = get_selection_added(selection, add, utterances)
  selection = get_selection_removed(selection, remove, utterances)
  assert selection == Selection(select)
  return select


class SelectTransformer():
  def __init__(self) -> None:
    self.add_transformer = AddTransformer()
    self.remove_tranformer = RemoveTransformer()

  def fit(self):
    self.add_transformer.fit()
    self.remove_tranformer.fit()

  def transform(self, selection: Selection, select: OrderedSet[UtteranceId], utterances: Utterances) -> Selection:
    remove = selection - select
    add = select - selection

    selection = self.add_transformer.transform(selection, add, utterances)
    selection = self.remove_tranformer.transform(selection, remove, utterances)
    assert selection == Selection(select)
    return selection
