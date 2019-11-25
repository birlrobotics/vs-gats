import gensim
from datasets.metadata import coco_classes
from datasets.hico_constants import HicoConstants
import os
import h5py
from tqdm import tqdm

#Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('datasets/word2vec/GoogleNews-vectors-negative300.bin', binary=True)  

data_const = HicoConstants()
original_keys = list(model.vocab.keys())
upper_keys = [str.upper(x) for x in original_keys]

hico_word2vec = os.path.join(data_const.proc_dir,'hico_word2vec.hdf5')
file = h5py.File(hico_word2vec, 'w')
for name in coco_classes:
    # if str.upper(name) not in upper_keys:
    #     print(name)
    #     continue
    if name == '__background__': continue

    if name == 'sports_ball':
        index = upper_keys.index(str.upper('footballs'))
        file.create_dataset(name, data=model[original_keys[index]])

    elif name == 'baseball_glove': 
        index = upper_keys.index(str.upper('mitt'))
        file.create_dataset(name, data=model[original_keys[index]])

    elif name == 'wine_glass': 
        index = upper_keys.index(str.upper('wine_glasses'))
        file.create_dataset(name, data=model[original_keys[index]])

    elif name == 'hair_drier': 
        index = upper_keys.index(str.upper('hair_driers'))
        file.create_dataset(name, data=model[original_keys[index]])
    else: 
        index = upper_keys.index(str.upper(name))
        file.create_dataset(name, data=model[original_keys[index]])

file.close()

# sports_ball == football
# baseball_glove == mitt
# wine_glass == wine_glasses
# hair_drier == hair_driers