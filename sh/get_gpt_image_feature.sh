pip install -U pip
apt-get install tmux
pip install connected-components-3d plyfile numba -i https://mirrors.aliyun.com/pypi/simple
cd /h3cstore_nt/JaneChen/flywire_NeuronRec/ || exit
CUDA_VISIBLE_DEVICES=0 nohup python main.py --config-base configs/Image-Base.yaml --config-file configs/imageEmbedding/Image-Unet-connector-extract_pc_gpt.yaml --inference >/h3cstore_nt/JaneChen/gpt-data/squence/log1.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python main.py --config-base configs/Image-Base.yaml --config-file configs/imageEmbedding/Image-Unet-connector-extract_pc_gpt.yaml --inference >/h3cstore_nt/JaneChen/gpt-data/squence/log2.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python main.py --config-base configs/Image-Base.yaml --config-file configs/imageEmbedding/Image-Unet-connector-extract_pc_gpt.yaml --inference >/h3cstore_nt/JaneChen/gpt-data/squence/log3.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 python main.py --config-base configs/Image-Base.yaml --config-file configs/imageEmbedding/Image-Unet-connector-extract_pc_gpt.yaml --inference
sleep 1h
CUDA_VISIBLE_DEVICES=0 python main.py --config-base configs/Image-Base.yaml --config-file configs/imageEmbedding/Image-Unet-connector-extract_pc_gpt.yaml --inference DATASET.OUTPUT_PATH /h3cstore_nt/JaneChen/gpt-data/squence/image-feature_with_corrs_and_candiadates/

# pip install yacs einops monai gunpowder cupy GPUtil
# pip install cilog fafbseg -i https://mirrors.aliyun.com/pypi/simple

#export PATH=/usr/local/cuda-11.3/bin:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH
#pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install -U numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip3 install fafbseg cilog protobuf==3.20.* ipywidgets -i https://pypi.tuna.tsinghua.edu.cn/simple
#export http_proxy="http://192.168.16.5:3128"
#apt install libgl1-mesa-glx -y
#cd /code || exit
#cp /code/sh/sharded.py /usr/local/lib/python3.8/dist-packages/cloudvolume/datasource/precomputed/skeleton/sharded.py