import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from datasets.metadata import coco_classes, action_classes
from datasets.hico_constants import HicoConstants
import os
import h5py
from tqdm import tqdm

# Load Google's pre-trained Word2Vec model.
# model_google = gensim.models.KeyedVectors.load_word2vec_format('/home/birl/ml_dl_projects/bigjun/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

# Convert GloVe vectors into the word2vec
glove_file = datapath('/home/birl/ml_dl_projects/bigjun/word2vec/glove.840B.300d.txt')
tmp_file = get_tmpfile("glove_840B_300d_word2vec.txt")
_ = glove2word2vec(glove_file, tmp_file)
model = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)

data_const = HicoConstants()
original_keys = list(model.vocab.keys())
upper_keys = [str.upper(x) for x in original_keys]

# word_list = open('/home/birl/ml_dl_projects/bigjun/word2vec/word_list.txt', 'w')
# for i in tqdm(range(len(original_keys))):
#     word_list.write(list(original_keys)[i]+'\n')
# word_list.close()
# import ipdb; ipdb.set_trace()
# hico_word2vec = os.path.join(data_const.proc_dir,'hico_word2vec.hdf5')

hico_word2vec = os.path.join(data_const.proc_dir,'hico_word2vec_include_verb.hdf5')
file = h5py.File(hico_word2vec, 'w')
for name in set(coco_classes+action_classes):
    # change the name to upper latter
    name_temp = str.upper(name)
    if '_' in name_temp:
        name_temp = name_temp.replace('_', '-')
    # if name_temp not in upper_keys:
    #     print(name)
    #     # import ipdb; ipdb.set_trace()
    #     continue
    if name == '__background__': continue

    # those not included in model should be replaced with similiar words
    if name == 'sports_ball':
        index = upper_keys.index(str.upper('footballs'))
        file.create_dataset(name, data=model[original_keys[index]])

    elif name == 'baseball_glove': 
        index = upper_keys.index(str.upper('mitt'))
        file.create_dataset(name, data=model[original_keys[index]])

    elif name == 'tennis_racket': 
        index = upper_keys.index(str.upper('racquet'))
        file.create_dataset(name, data=model[original_keys[index]])

    elif name == 'potted_plant': 
        index = upper_keys.index(str.upper('bonsai'))
        file.create_dataset(name, data=model[original_keys[index]])

    elif name == 'brush_with': 
        index1 = upper_keys.index(str.upper('brush'))
        index2 = upper_keys.index(str.upper('with'))
        file.create_dataset(name, data=(model[original_keys[index1]]+model[original_keys[index2]])/2)
        file.create_dataset(name+'_no_everage', data=model[original_keys[index1]]+model[original_keys[index2]])

    elif name == 'cut_with': 
        index1 = upper_keys.index(str.upper('cut'))
        index2 = upper_keys.index(str.upper('with'))
        file.create_dataset(name, data=(model[original_keys[index1]]+model[original_keys[index2]])/2)

    elif name == 'drink_with': 
        index1 = upper_keys.index(str.upper('drink'))
        index2 = upper_keys.index(str.upper('with'))
        file.create_dataset(name, data=(model[original_keys[index1]]+model[original_keys[index2]])/2)
        file.create_dataset(name+'_no_everage', data=model[original_keys[index1]]+model[original_keys[index2]])

    elif name == 'eat_at': 
        index1 = upper_keys.index(str.upper('eat'))
        index2 = upper_keys.index(str.upper('at'))
        file.create_dataset(name, data=(model[original_keys[index1]]+model[original_keys[index2]])/2)
        file.create_dataset(name+'_no_everage', data=model[original_keys[index1]]+model[original_keys[index2]])

    elif name == 'lie_on': 
        index = upper_keys.index(str.upper('lie-down'))
        file.create_dataset(name, data=model[original_keys[index]])

    elif name == 'no_interaction': 
        index1 = upper_keys.index(str.upper('no'))
        index2 = upper_keys.index(str.upper('interaction'))
        file.create_dataset(name, data=(model[original_keys[index1]]+model[original_keys[index2]])/2)
        file.create_dataset(name+'_no_everage', data=model[original_keys[index1]]+model[original_keys[index2]])

    elif name == 'sit_at': 
        index1 = upper_keys.index(str.upper('sit'))
        index2 = upper_keys.index(str.upper('at'))
        file.create_dataset(name, data=(model[original_keys[index1]]+model[original_keys[index2]])/2)
        file.create_dataset(name+'_no_everage', data=model[original_keys[index1]]+model[original_keys[index2]])

    elif name == 'stand_under': 
        index1 = upper_keys.index(str.upper('stand'))
        index2 = upper_keys.index(str.upper('under'))
        file.create_dataset(name, data=(model[original_keys[index1]]+model[original_keys[index2]])/2)
        file.create_dataset(name+'_no_everage', data=model[original_keys[index1]]+model[original_keys[index2]])

    elif name == 'stop_at': 
        index1 = upper_keys.index(str.upper('stop'))
        index2 = upper_keys.index(str.upper('at'))
        file.create_dataset(name, data=(model[original_keys[index1]]+model[original_keys[index2]])/2)
        file.create_dataset(name+'_no_everage', data=model[original_keys[index1]]+model[original_keys[index2]])

    elif name == 'talk_on': 
        index1 = upper_keys.index(str.upper('talk'))
        # index2 = upper_keys.index(str.upper('on'))
        # file.create_dataset(name, data=(model[original_keys[index1]]+model[original_keys[index2]])/2)
        file.create_dataset(name, data=model[original_keys[index1]])

    elif name == 'text_on': 
        index1 = upper_keys.index(str.upper('text'))
        # index2 = upper_keys.index(str.upper('on'))
        # file.create_dataset(name, data=(model[original_keys[index1]]+model[original_keys[index2]])/2)
        file.create_dataset(name, data=model[original_keys[index1]])

    elif name == 'type_on': 
        index1 = upper_keys.index(str.upper('type'))
        # index2 = upper_keys.index(str.upper('on'))
        # file.create_dataset(name, data=(model[original_keys[index1]]+model[original_keys[index2]])/2)
        file.create_dataset(name, data=model[original_keys[index1]])

    else: 
        try:
            index = upper_keys.index(name_temp)
            file.create_dataset(name, data=model[original_keys[index]])
        except Exception as e:
            import ipdb; ipdb.set_trace()
            print(e)

file.close()

# model_google
# sports_ball == football
# baseball_glove == mitt
# wine_glass == wine_glasses
# hair_drier == hair_driers
# cut_with
# drink_with
# eat_at
# lie_on
# no_interaction
# sit_at
# sit_on
# stand_under
# stop_at
# talk_on
# text_on