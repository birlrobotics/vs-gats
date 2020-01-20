mkdir datasets/processed
python -m datasets.vcoco_run_faster_rcnn
python -m datasets.vcoco_select_confident_boxes
python -m datasets.vcoco_word2vec
python -m datasets.vcoco_train_val_test_data 
python -m datasets.vcoco_spatial_feature