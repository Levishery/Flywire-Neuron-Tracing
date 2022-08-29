export https_proxy=192.168.16.5:3128
export PATH=/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install fafbseg cilog ipywidgets -i https://pypi.tuna.tsinghua.edu.cn/simple
cd /braindat/lab/liusl/flywire || exit
mv sh/sharded.py /opt/conda/lib/python3.8/site-packages/cloudvolume/datasource/precomputed/skeleton/sharded.py

