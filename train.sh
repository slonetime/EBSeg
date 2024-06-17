export DETECTRON2_DATASETS=path_to_datasets_dir
python train_net.py \
 --config-file configs/ebseg/ebseg_b.yaml \
 --num-gpus 4 \
 OUTPUT_DIR path_to_output_dir