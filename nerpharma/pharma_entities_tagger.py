import textdistance
import pandas as pd
from .ner_tagger import NERAutoTagger

class PharmaEntitiesTagger(NERAutoTagger):

    """
        Tagging Pharmaceutical companies in a text
    """

    def __init__(self, entity, path_to_entities_dataset, path_to_countries_dataset, path_to_eng_words_dataset):
        super(PharmaEntitiesTagger, self).__init__(entity, path_to_entities_dataset)
        self.__init_data(path_to_countries_dataset, path_to_eng_words_dataset)

    def __init_data(self, path_to_countries_dataset, path_to_eng_words_dataset):
        f = open(path_to_eng_words_dataset, "r")
        self.eng_words = f.read().split()
        countries_df = pd.read_csv(path_to_countries_dataset)
        self.all_countries = [l.strip().title()
                              for l in countries_df.Country.to_list()]

    def _is_detected_word_an_entity(self, word):
        is_pc = False
        for company in self.all_entities:
            if (word[0].isupper() and not word[1].isupper() and textdistance.cosine.normalized_similarity(company.lower(), word.lower()) > 0.9):
                is_pc = True
                break
            if (word[0].isupper() and not word[1].isupper() and textdistance.cosine.normalized_similarity(company.lower(), word.lower()) > 0.5 and company.lower().find(word.lower()) >= 0):
                is_pc = True
                break
            if(word[0].isupper() and word.title() in self.task_specific_keywords):
                is_pc = True
                break
        return is_pc

    def _filter_task_specific_keywords(self, df):
        df_copy = df.copy()
        for index in df.index:
            if(df_copy.loc[index].token in self.task_specific_keywords or df_copy.loc[index].token in self.all_countries or df_copy.loc[index].token.lower() in self.eng_words):
                df_copy = df_copy.drop(index)

        return df_copy
