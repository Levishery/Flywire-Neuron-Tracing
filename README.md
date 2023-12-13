# flywire_NeuronRec
## Dataset Access
### FlyTracing pairwise segment connection dataset
The dataset and our best pre-trained models is available at [Google Drive](https://drive.google.com/drive/folders/1FPg8q8BE-R-BiVxdAZ8qHDYLacYGLxgJ?usp=drive_link)
### FAFB EM image
To download the EM image blocks:
```bash
cd dataset/download_fafb.py
# edit the block name csv path (provided in the link above) and destination path to yours
python download_fafb.py
```
The 4000 blocks require about 1TB of storage space.
## Fintune the models on SNEMI3D
### Input: 

training -- raw image, gt segmentation, initial over-segmentation;

testing -- raw image, initial over-segmentation
### Steps: 
1. Prepare image patches of positive connections for finetuning Connect-Embed using [get_snemi_patch.py](https://github.com/Levishery/Biological-graph/blob/main/biologicalgraphs/neuronseg/scripts/get_snemi_patch.py);
   The code produces patch image/GT/segmentation centered at the center of gravity of the connected area between segment pairs;
2. Sample point cloud from segmentation [get_pc_snemi3d.py](https://github.com/Levishery/flywire_NeuronRec/blob/main/dataset/snemi3d/get_pc_snemi3d.py);
3. Fintune the image embedding model and run inference on the test set [config](https://github.com/Levishery/flywire_NeuronRec/blob/main/configs/imageEmbedding/Image-Unet-SNEMI3D.yaml);
4. Map the computed embedding to the point cloud [map_pc_snemi3d.py](https://github.com/Levishery/flywire_NeuronRec/blob/main/dataset/snemi3d/map_pc_snemi3d.py);
5. Finetune the Pointnet++ and infer

TODO: merge step 3 and 4
