cd /code || exit
python main.py --config-base configs/Image-Base.yaml --config-file configs/Image-swin.yaml
cp -r /output/logs /braindat/lab/liusl/flywire/experiment/swin-cremi-metric