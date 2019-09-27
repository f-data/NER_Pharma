
class SpaCyTagger:

    def __init__(self, entity):
        self.entity = entity

    def get_spacy_tokens(self, tokenized_cleaned, sentence):
        """
            Return SpaCy training tuple based on the tokenized sentece
            tokenized_cleaned: A dataframe that contains a `range` column with the entitiy range in the sentece (start:end)
        """
        entities = []

        for t in tokenized_cleaned.itertuples():
            entities.append((int(t.range.split(":")[0]), int(t.range.split(":")[1]), self.entity))
            
        return (sentence, {"entities" : entities})
