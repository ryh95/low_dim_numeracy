
# load sentences
import pickle

with open('sel_number_sens.pickle','rb') as f:
    sens = pickle.load(f)

# lemmatize the sentences
from collections import deque

import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma',tokenize_pretokenized=True)
doc = nlp(list(sens))
lemma_sen = deque()
for sen in doc.sentences:
    lemma_sen.append([word.lemma for word in sen.words])

# remove stopwords
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))
# processed_sen = deque()
# for sen in lemma_sen:
#     processed_sen.append([word for word in sen if word not in stop_words])

# save processed sentences
with open('processed_sentences_keep_stopwords.pickle','wb') as f:
    pickle.dump(lemma_sen,f,pickle.HIGHEST_PROTOCOL)