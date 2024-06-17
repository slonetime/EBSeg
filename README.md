# [CVPR2024] Open-Vocabulary Semantic Segmentation with Image Embedding Balancing

We release our code and trained models for our CVPR2024 paper [Open-Vocabulary Semantic Segmentation with Image Embedding Balancing](https://openaccess.thecvf.com/content/CVPR2024/html/Shan_Open-Vocabulary_Semantic_Segmentation_with_Image_Embedding_Balancing_CVPR_2024_paper.html)

## Getting started
### Environment setup

First, clone this repo:
``` bash
git clone https://github.com/slonetime/EBSeg.git
```
Then, create a new conda env and install required packeges:
``` bash
cd EBSeg
conda create --name ebseg python=3.9
conda activate ebseg
pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
At last, install the MultiScaleDeformableAttention in Mask2former:

``` bash
cd ebseg/model/mask2former/modeling/pixel_decoder/ops/
sh make.sh 
```

### Data preparation

We follow the dataset preparation process in SAN, so please follow the instructions in https://github.com/MendelXu/SAN?tab=readme-ov-file#data-preparation.

## Training

First, change the config_file path, dataset_dir path and ourput_dir path in train.sh. Then, you can train an EBSeg model with the following command:
``` bash
bash train.sh
```

## Inference with our trained model

Download our trained models from the url links in the followding table(with mIoU metric):

| Model | A-847 | PC-459| A-150| PC-59| VOC|
|---|---|---|---|---|---|
|[EBSeg-B](https://huggingface.co/slonetime/EBSeg/resolve/main/EBSeg_base.pth) | 11.1 | 17.3 | 30.0 | 56.7 | 94.6 |
|[EBSeg-L](https://huggingface.co/slonetime/EBSeg/resolve/main/EBSeg_large.pth)| 13.7 | 21.0 | 32.8 | 60.2 | 96.4 |


Like training, you should change the config_file path, dataset_dir path, checkpoint path and ourput_dir path in test.sh. Then, test a EBSeg model by:
``` bash
bash test.sh
```

## Acknowledgments

Our code are based on [SAN](https://github.com/MendelXu/SAN), [CLIP](https://github.com/openai/CLIP), [CLIP Surgery](https://github.com/xmed-lab/CLIP_Surgery), [Mask2former](https://github.com/facebookresearch/Mask2Former) and [ODISE](https://github.com/NVlabs/ODISE).

We thanks them for their excellent works!
