from datetime import datetime
import functools
import numpy as np
import pandas as pd
import pickle
import time
from requests import Request, Session
import json
import math
import naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# Load pretrained model and vectorizer
f = open('nets/configs/naive_bayes.pickle', 'rb')
clf = pickle.load(f)
f.close()

# Begin Validation Test on Non-English Languages
valid = pd.read_csv('data/validation.csv')

# Translation from src lang -> en
def prep_translate(text: list, lang: str):
    url = 'https://translate.yandex.net/api/v1.5/tr.json/translate'
    with open('nets/yandex_api_key.txt') as f:
        k =  f.readlines()[0]
    f.close()
    j = {
        'text': text,
        'lang': lang,
        'format': 'plain',
        'key': k
    }
    req = Request('POST', url, data=j)
    prepped = req.prepare()
    prepped.headers['Content-Length'] = str(len(prepped.body.encode('ascii')))
    return prepped


def send_translation(preps):
    s = Session()
    responses = list(map(s.send, preps))
    try:
        return [r for response in responses for r in response.json()['text']]
    except KeyError:
        print('Errors: ', len([r.status_code for r in responses if r.status_code != 200]))
        print('Error Message: ', [r.json() for r in responses if r.status_code != 200][0])



def index_marks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)

valid['translated'] = ''
dfs = [valid.loc[valid['lang']==lang] for lang in pd.unique(valid['lang'])]
dfs = [df.reset_index() for df in dfs]

for df in dfs:
    chunks = [d for d in np.split(df, index_marks(df.shape[0], 20)) if len(d) > 0]
    texts = map(functools.partial(prep_translate, lang=df.loc[0,'lang']), \
                [chunk['comment_text'].values.tolist() for chunk in chunks])
    response = send_translation(texts)
    break
    print('Sleeping for 100 seconds...to prevent "exceeding user limit"...')
    df['translated'] = response
    time.sleep(100)
    print('I am awake !')

# Data preprocessing and removing 'useless' columns
df = pd.concat(dfs)
df.sort_values(by=['id'], inplace=True)
df['translated'] = df['translated'].apply(lambda x: naive_bayes.data_preprocessing(x))
df['is_toxic'] = 0
df.drop(['index', 'comment_text', 'lang'], axis=1, inplace=True)

# Text Vectorization
predict = clf.predict(df['translated'].tolist())
print(f'Accuracy on Non-English: {accuracy_score(predict, df["toxic"])}')
df['toxic'] = predict

# Exporting Results to CSV
df.drop([v for v in df.columns if v not in ['id', 'toxic']], axis=1, inplace=True)
# Generate time stamp while saving
date = datetime.now()
timestamp = date.strftime('%Y%m%d-%H:%M')
df.to_csv(f'nets/submissions/sub{timestamp}.csv', index=False)
