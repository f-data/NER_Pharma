import textdistance
import pandas as pd
from .ner_tagger import NERAutoTagger


class DrugEntitiesTagger(NERAutoTagger):

    """
        Tagging Drugs in text
    """

    def __init__(self, entity, path_to_entities_dataset):
        super(DrugEntitiesTagger, self).__init__(entity, path_to_entities_dataset)

    def _is_detected_word_an_entity(self, word):
        is_pc = False
        for drug in self.all_entities:
            if(word == drug):
                is_pc = True
                break
        return is_pc

    def _filter_task_specific_keywords(self, df):
        df_copy = df.copy()
        return df_copy
