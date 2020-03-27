from os.path import join

from config import EMB_DIR
from utils import extract_num_emb

# preprocess_google_news_skip(join(EMB_DIR,'GoogleNews-vectors-negative300'))
extract_num_emb(join(EMB_DIR, 'skipgram-5.txt'))