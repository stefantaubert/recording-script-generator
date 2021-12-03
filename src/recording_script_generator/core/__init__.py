from recording_script_generator.core.detection import (
    select_utterances_through_greedy_duration_inplace,
    select_utterances_through_kld_duration_inplace)
from recording_script_generator.core.inspection import (
    remove_deselected_utterances, remove_duplicate_utterances_inplace,
    remove_from_selection_inplace, remove_from_utterances_inplace,
    remove_utterances_with_acronyms_inplace,
    remove_utterances_with_custom_word_counts,
    remove_utterances_with_non_dictionary_words,
    remove_utterances_with_proper_names, remove_utterances_with_undesired_text,
    remove_utterances_with_unfrequent_words)
