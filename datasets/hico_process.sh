mkdir datasets/processed
python -m datasets.hico_mat_to_json
python -m datasets.hico_hoi_cls_count
python -m datasets.hico_split_ids
python -m datasets.hico_run_faster_rcnn
python -m datasets.hico_select_confident_boxes
python -m datasets.evaluate_instance_detection
python -m datasets.hico_word2vec
python -m datasets.hico_train_val_test_data --f_t='fc7'
python -m datasets.hico_spatial_feature