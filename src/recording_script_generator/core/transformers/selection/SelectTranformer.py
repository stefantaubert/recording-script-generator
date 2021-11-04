from ordered_set import OrderedSet
from recording_script_generator.core.transformers.selection.AddTransformer import \
    AddTransformer
from recording_script_generator.core.transformers.selection.RemoveTransformer import \
    RemoveTransformer
from recording_script_generator.core.types import (Selection, UtteranceId,
                                                   Utterances)


class SelectTranformer():
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
