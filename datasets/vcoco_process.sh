mkdir datasets/processed
python -m datasets.vcoco_run_faster_rcnn
python -m datasets.vcoco_word2vec
python -m datasets.vcoco_select_confident_boxes
python -m datasets.vcoco_train_val_test_data 
python -m datasets.vcoco_spatial_feature
python -m datasets.vcoco_trainval_data
# python -m vcoco_train --e_v='HICO_trainval' \
#                       --t_m='epoch' --b_s=32 --f_t='fc7' --layers=1 \
#                       --bn=False --lr=0.00001 \
#                       --drop_prob=0.5  --m_a='false' \
#                       --d_a='false' --bias='true' --optim='adam' \
#                       --diff_edge='false' --epoch=1500  \
#                       --hico='/home/birl/ml_dl_projects/bigjun/hoi/VS_GATs/result/checkpoint_248_epoch.pth'