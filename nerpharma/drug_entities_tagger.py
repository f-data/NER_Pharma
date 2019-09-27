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
            if (word[0].isupper() and not word[1].isupper() and textdistance.cosine.normalized_similarity(drug.lower(), word.lower()) > 0.9):
                is_pc = True
                break
            if (word[0].isupper() and not word[1].isupper() and textdistance.cosine.normalized_similarity(drug.lower(), word.lower()) > 0.5 and drug.lower().find(word.lower()) >= 0):
                is_pc = True
                break
            if(word[0].isupper() and word.title() in self.task_specific_keywords):
                is_pc = True
                break
        return is_pc

    def _filter_task_specific_keywords(self, df):
        df_copy = df.copy()
        return df_copy
