import gensim
from gensim.test.utils import datapath
from gensim import utils
import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from pprint import pprint
import tempfile

df = pd.read_csv("jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

processed_df = pd.DataFrame(columns=["comment", "toxic"])

processed_df[["toxic"]] = pd.DataFrame(df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum(axis=1))
processed_df[["toxic"]] = processed_df[["toxic"]].mask(processed_df[["toxic"]] >= 1, 1)

#borrowed some of your code
stopwords = stopwords.words('english')
stopwords.remove('not')
lemmatizer = WordNetLemmatizer()

def data_preprocessing(comment):
    comment = re.sub(re.compile(r'\''), '', comment) # removing contractions
    comment = re.sub(re.compile(r'[^A-Za-z]+'), ' ', comment) # only words
    comment = comment.lower()
    tokens = word_tokenize(comment)

    # stop words removal
    comment = [word for word in tokens if word not in stopwords]
    # lemmatizing words
    comment = [lemmatizer.lemmatize(word) for word in comment]
    return ' '.join(comment)

processed_df[["comment"]] = df['comment_text'].apply(lambda x: data_preprocessing(x))

____
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('lee_background.cor')
        for line in range(len(processed_df)):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(processed_df.at[line, "comment"])


path = get_tmpfile("../input/word2vec.model")

sentences = MyCorpus()
print(sentences)
model = gensim.models.Word2Vec(sentences=sentences, size=50, callbacks=[callback()])
model.save("word2vec.model")


new_model = gensim.models.Word2Vec.load(temporary_filepath)
