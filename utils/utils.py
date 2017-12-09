import numpy as np
import _pickle as cPickle
import os

def load_pickle(path):
    with open(path, 'rb') as f:
        file = cPickle.load(f)
        print ('Loaded %s..' %path)
        return file

def save_pickle(data, path):
    with open(path, 'wb') as f:
        cPickle.dump(data, f)
        print ('Saved %s..' %path)

def load_dataset(split):
    data_path = os.path.join('./dataset', split)
    data = {}

    # Load Preprocessed Data
    data['features'] = load_pickle(os.path.join(data_path, '%s_features.hkl' % split))
    data['file_names'] = load_pickle(os.path.join(data_path, '%s_filenames.pkl' % split))
    data['captions'] = load_pickle(os.path.join(data_path, '%s_captions.pkl' % split))
    data['image_idxs'] = load_pickle(os.path.join(data_path, '%s_imageIdxs.pkl' % split))

    if split == 'train':
        data['word2idx'] = load_pickle(os.path.join(data_path, 'vocab.pkl'))

    return data
