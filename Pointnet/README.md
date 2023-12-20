# Segment connectivity prediction with PointNet++ 
Original implementation of [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) cloned from [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
### Fintuning on SNEMI3D
'''
# first prepare the point clouds and image embeddings following the instruction in ../README.md
# prepare data names
python data_utils/get_snemi3d_names.py
python train_classification_wImage_snemi.py --model pointnet2_binary_ssg --log_dir connection_snemi3d --num_gpus 1 --batch_size 96 --learning_rate 0.0005 --num_point 2048 --restart --image_feature connect-embed-best/ --checkpoint pointnet_best_model.pth
'''



