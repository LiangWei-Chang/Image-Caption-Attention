import sys

sys.path.append('utils')
sys.path.append('model')

from preprocess_data import build_vocab
from utils import *

annotations = load_pickle('./dataset/train_annotations.pkl')
word2idx, idx2word = build_vocab(annotations=annotations, threshold=30)
save_pickle(word2idx, './dataset/word2idx.pkl')
save_pickle(idx2word, './dataset/idx2word.pkl')
