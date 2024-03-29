# flywire_NeuronRec
Learning Multimodal Volumetric Features for Large-Scale Neuron Tracing
https://arxiv.org/abs/2401.03043
## Dataset Access
### FlyTracing pairwise segment connection dataset
The dataset and our best pre-trained models are available at [Google Drive](https://drive.google.com/drive/folders/1FPg8q8BE-R-BiVxdAZ8qHDYLacYGLxgJ?usp=drive_link)
### FAFB EM image
To download the EM image blocks:
```bash
cd dataset/download_fafb.py
# edit the block name csv path (provided in the link above) and destination path to yours
python download_fafb.py
```
The 4000 blocks require about 1TB of storage space.
## Environment 
Most of the dependencies are included in this docker image:
```
From registry.cn-hangzhou.aliyuncs.com/janechen/xp_projects:v1
```
Extra packages:
```bash
pip install connected-components-3d plyfile numba 
```
## Fintune the models on SNEMI3D
### Input: 

training -- raw image, gt segmentation, initial over-segmentation;

testing -- raw image, initial over-segmentation
### Steps: 
1. Prepare image patches of positive connections for finetuning Connect-Embed using [get_snemi_patch.py](https://github.com/Levishery/Biological-graph/blob/main/biologicalgraphs/neuronseg/scripts/get_snemi_patch.py);
   The code produces patch image/GT/segmentation centered at the center of gravity of the connected area between segment pairs;
2. Sample point cloud from segmentation [get_pc_snemi3d.py](https://github.com/Levishery/flywire_NeuronRec/blob/main/dataset/snemi3d/get_pc_snemi3d.py);
3. Fintune the image embedding model and run inference on the test set [config](https://github.com/Levishery/flywire_NeuronRec/blob/main/configs/imageEmbedding/Image-Unet-SNEMI3D.yaml)
```bash
# train
python main.py --config-base configs/Image-Base.yaml --config-file configs/imageEmbedding/Image-Unet-SNEMI3D.yaml --checkpoint embedding_best_model.pth
# inference
python main.py --config-base configs/Image-Base.yaml --config-file configs/imageEmbedding/Image-Unet-SNEMI3D.yaml --checkpoint SNEMI_embedding_best.pth --inference INFERENCE.OUTPUT_PATH test SYSTEM.NUM_CPUS 0
```
4. Map the computed embedding to the point cloud [map_pc_snemi3d.py](https://github.com/Levishery/flywire_NeuronRec/blob/main/dataset/snemi3d/map_pc_snemi3d.py);
5. Finetune the Pointnet++ refer to [Pointnet/README](https://github.com/Levishery/Flywire-Neuron-Tracing/tree/main/Pointnet)

TODO: merge step 3 and 4

### Citation
If you find our repository useful in your research, please consider citing:
```bibtex
@inproceedings{chen2024learning,
      title={Learning Multimodal Volumetric Features for Large-Scale Neuron Tracing}, 
      author={Qihua Chen and Xuejin Chen and Chenxuan Wang and Yixiong Liu and Zhiwei Xiong and Feng Wu},
      year={2024},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
}
```
