import string
from nltk.corpus import stopwords
from collections import OrderedDict
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import words
import nltk
import itertools
from abc import ABC, abstractmethod
from .external.ner_to_spacy import SpaCyTagger

class NERAutoTagger(ABC):
    """
        Perform auto tagging for entities in a text based on provided dataset with entities
    """

    task_specific_keywords = []

    def __init__(self, entity, path_to_entities_dataset):
        self.entity = entity
        self.entities_df = pd.read_csv(path_to_entities_dataset)

        self.cachedStopWords = stopwords.words("english")
        self.__init_entities()
        self.spacy_tagger = SpaCyTagger(self.entity)

    def __init_entities(self):
        self.all_entities = [l.strip().title()
                             for l in self.entities_df.Text.to_list()]

    def __find_consecutive_indexes(self, indexes, currentIndex):
        """
        Retun a list of consecutive tokens (distance  <= 2 due to punctuation or spaces) to the current token with provided index
        """
        consecutive_indexes = []
        
        for i in indexes:
            if(i > currentIndex):
                if(len(consecutive_indexes) == 0):
                    if(i - currentIndex <= 2):
                        consecutive_indexes.append(i)
                elif(i - consecutive_indexes[-1] <= 2):
                    consecutive_indexes.append(i)
        
        return consecutive_indexes

    @abstractmethod
    def _is_detected_word_an_entity(self, word):
        """
            Task specific, should be implemented by the concrete component that inherrits the class
        """
        raise NotImplementedError

    @abstractmethod
    def _filter_task_specific_keywords(self, df):
        """
            Task specific, should be implemented by the concrete component that inherrits the class
        """
        raise NotImplementedError

    def __entities_tokenized_cleaned(self, tokenized_news, should_merge = True):
        tokenized_news = tokenized_news[tokenized_news.token != " "]
        tokenized_news = tokenized_news[~tokenized_news.token.str.lower().isin(self.cachedStopWords)]
        tokenized_news = tokenized_news[tokenized_news.token.str.len() > 1]

        for i in tokenized_news.index:
            if not self._is_detected_word_an_entity(tokenized_news.loc[i].token):
                tokenized_news = tokenized_news[tokenized_news.index != i]

        if(not should_merge):
            return self._filter_task_specific_keywords(tokenized_news)

        tokenized_news_copy = tokenized_news.copy()

        all_tokenized_indexes = tokenized_news_copy.index
        for index in all_tokenized_indexes:
            # Find consecutive indexes
            if(index not in tokenized_news_copy.index):
                continue
            consecutive_indexes = self.__find_consecutive_indexes(all_tokenized_indexes, index)
            if(len(consecutive_indexes) > 0):
                # Append to current
                # Remove from dataframe
                for c in consecutive_indexes:
                    if(c not in tokenized_news_copy.index):
                        continue
                    tokenized_news_copy.loc[index].token += " "
                    tokenized_news_copy.loc[index].token += tokenized_news_copy.loc[c].token
                    # Merge start-end
                    range1 = tokenized_news_copy.loc[index].range
                    range2 = tokenized_news_copy.loc[c].range
                    tokenized_news_copy.loc[index].range = range1.split(":")[0] + ":" + range2.split(":")[1]
                    tokenized_news_copy = tokenized_news_copy.drop(c)

            tokenized_news_copy = self._filter_task_specific_keywords(
                tokenized_news_copy)

        return tokenized_news_copy

    def __index_sentence_tokens(self, tokenized_sentence, start = 0):
        index = 0
        tokenized_df = pd.DataFrame(columns=['token', 'range'])
        start = start
        for token in tokenized_sentence:
            tokenized_df.loc[index] = [token, str(start) + ":" + str(start + len(token))]
            start += len(token)
            index += 1
            
        return tokenized_df

    def __get_tokenized_entities_cleaned(self, tokeninzed_sent):
            intermediate_tokenized_sentence = [[word_tokenize(w), ' '] for w in tokeninzed_sent.split()]
            intermediate_tokenized_sentence = list(itertools.chain(*list(itertools.chain(*intermediate_tokenized_sentence))))
            intermediate_tokenized_sentence = intermediate_tokenized_sentence[0:len(intermediate_tokenized_sentence) - 1]
            df_tok = self.__index_sentence_tokens(intermediate_tokenized_sentence)
            return self.__entities_tokenized_cleaned(df_tok)

    def __tokenize_full_text_SpaCy(self, text):
        """
            Create a dataset for training a SpaCy model
        """
        tokenized_sentences = sent_tokenize(text)
        spacy_tokens = []
        for index, t in enumerate(tokenized_sentences):
            tokenized_cleaned = self.__get_tokenized_entities_cleaned(t)
            spacy_tokens.append(self.spacy_tagger.get_spacy_tokens(tokenized_cleaned, t))
        return spacy_tokens

    def __tokenize_full_text_allenNLP(self, text):
        tokenized_text_df = pd.DataFrame(columns=['token', 'range'])
        
        tokenized_sentences = sent_tokenize(text)
        
        text_start = 0
        for index, t in enumerate(tokenized_sentences):
            intermediate_tokenized_sentence = [[word_tokenize(w), ' '] for w in t.split()]
            intermediate_tokenized_sentence = list(itertools.chain(*list(itertools.chain(*intermediate_tokenized_sentence))))
            intermediate_tokenized_sentence = intermediate_tokenized_sentence[0:len(intermediate_tokenized_sentence) - 1]
            df_tok = self.__index_sentence_tokens(intermediate_tokenized_sentence, text.find(t))
            tokenized_text_df = pd.concat([tokenized_text_df, df_tok], ignore_index=True)
            text_start += len(t)

        cleaned_df = self.__entities_tokenized_cleaned(tokenized_text_df, False)
        cleaned_df["tag"] = "O"
        tokenized_text_df = tokenized_text_df[tokenized_text_df.token != " "]
        tokenized_text_df["tag"] = "O"

        for index in tokenized_text_df.index:
            if(index in cleaned_df.index and tokenized_text_df.loc[index].tag == "O"):
                consecutive_indexes = self.__find_consecutive_indexes(
                    cleaned_df.index, index)

                if(len(consecutive_indexes) == 0):
                    # U
                    tokenized_text_df.loc[index].tag = "U-" + self.entity
                else:
                    # BIL
                    tokenized_text_df.loc[index].tag = "B-" + self.entity
                    for c in consecutive_indexes:
                        tokenized_text_df.loc[c].tag = "I-" + self.entity
                    tokenized_text_df.loc[consecutive_indexes[len(consecutive_indexes) - 1]].tag = "L-" + self.entity
            
        return tokenized_text_df[['token', 'tag']]

    def __tokenized_full_text_Default(self, text):
        tokenized_text_df = pd.DataFrame(columns=['token', 'range'])
        
        tokenized_sentences = sent_tokenize(text)
        text_start = 0
        for index, t in enumerate(tokenized_sentences):
            intermediate_tokenized_sentence = [[word_tokenize(w), ' '] for w in t.split()]
            intermediate_tokenized_sentence = list(itertools.chain(*list(itertools.chain(*intermediate_tokenized_sentence))))
            intermediate_tokenized_sentence = intermediate_tokenized_sentence[0:len(intermediate_tokenized_sentence) - 1]
            df_tok = self.__index_sentence_tokens(intermediate_tokenized_sentence, text.find(t))
            tokenized_text_df = pd.concat([tokenized_text_df, df_tok], ignore_index=True)
            text_start += len(t)

        return self.__entities_tokenized_cleaned(tokenized_text_df)

    def update_task_specific_keywords(self, keywords):
        """
            Add task specific keywords to be filtered out while tagging entities. Eg. Pharmacy based keywords may be: Medical, Pharmaceuticals, Health etc
        """
        self.task_specific_keywords = keywords

    def tokenize_full_text(self, text, ner_type = "default"):
        """
            Generate tagged data for the specified entity
        """
        if(ner_type == "spacy"):
            return self.__tokenize_full_text_SpaCy(text)
        elif(ner_type == "allen"):
            return self.__tokenize_full_text_allenNLP(text)
        else:
            return self.__tokenized_full_text_Default(text)
 