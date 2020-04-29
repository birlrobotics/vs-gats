import os
import time
import h5py
from datasets.vcoco_constants import VcocoConstants

def combine_visual_data(save_data, data1, data2):
    for id in data1.keys():
        save_data.create_group(id)
        for key, value in data1[id].items():
            save_data[id].create_dataset(key, data=value)

    for id in data2.keys():
        save_data.create_group(id)
        for key, value in data2[id].items():
            save_data[id].create_dataset(key, data=value)

def combine_spatial_data(save_data, data1, data2):
    for key, value in data1.items():
        save_data.create_dataset(key, data=value)
            
    for key, value in data2.items():
        save_data.create_dataset(key, data=value)

def main(data_const, subset='vcoco_trainval'):
    '''
    combine train && val data into trainval data set
    '''
    # create the file to save the data
    subset_path = os.path.join(data_const.proc_dir, subset)
    if not os.path.exists(subset_path):
        os.makedirs(subset_path)
    visual_hdf5 = os.path.join(subset_path, 'vcoco_data.hdf5')
    spatial_hdf5 = os.path.join(subset_path, 'spatial_feat.hdf5')
    visual_data = h5py.File(visual_hdf5, 'w')
    spatial_data = h5py.File(spatial_hdf5, 'w')
    # load data from train && val set
    t_v_d = h5py.File(data_const.train_visual_data, 'r')
    t_s_d = h5py.File(data_const.train_spatial_data, 'r')
    v_v_d = h5py.File(data_const.val_visual_data, 'r')
    v_s_d = h5py.File(data_const.val_spatial_data, 'r')
    # start combining
    print('start combining')
    start_time = time.time()
    # import ipdb; ipdb.set_trace()
    combine_visual_data(visual_data, t_v_d, v_v_d)
    combine_spatial_data(spatial_data, t_s_d, v_s_d)
    print('Finish combining. Spend time {}'.format(time.time()-start_time))
    visual_data.close()
    spatial_data.close()

if __name__ == '__main__':
    data_const = VcocoConstants()
    main(data_const)