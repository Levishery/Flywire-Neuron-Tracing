cd /code || exit
# python main.py --config-base configs/Image-Base.yaml --config-file configs/Image-swin-embeddingonly.yaml
python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main.py --distributed --config-base configs/Image-Base.yaml --config-file configs//Image-swin-embeddingonly.yaml