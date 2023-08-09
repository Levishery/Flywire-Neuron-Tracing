export https_proxy=192.168.16.5:3128
cp .sh/sharded.py /usr/local/lib/python3.8/dist-packages/cloudvolume/datasource/precomputed/skeleton/sharded.py
pip install -U pip
apt-get install tmux
pip install connected-components-3d
pip install plyfile
python main.py --config-base configs/Image-Base.yaml --config-file configs/imageEmbedding/Image-Unet-connector-extract_pc_test.yaml --inference &
sleep 6000s
python main.py --config-base configs/Image-Base.yaml --config-file configs/imageEmbedding/Image-Unet-connector-extract_pc_test.yaml --inference