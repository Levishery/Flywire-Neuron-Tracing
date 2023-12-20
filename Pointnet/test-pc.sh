cd /braindat/lab/liusl/flywire/PointNet/Pointnet_Pointnet2_pytorch-master/ || exit
pip install plyfile
python test_classification_wImage.py --model pointnet2_binary_ssg --log_dir connection_lr0005_bs92_2048  --num_gpus 1 --batch_size 92 --learning_rate 0.0005 --num_point 2048