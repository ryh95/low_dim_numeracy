from os.path import join

from config import EMB_DIR
from utils import preprocess_google_news_skip, preprocess_skipgram2

# preprocess_google_news_skip(join(EMB_DIR,'GoogleNews-vectors-negative300'))
preprocess_skipgram2(join(EMB_DIR,'crawl-300d-2M-subword'))