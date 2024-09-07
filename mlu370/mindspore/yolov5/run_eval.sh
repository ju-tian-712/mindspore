# Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID]
# bash run_eval.sh ../coco2017_1000/ ../coco2017_1000/weight.ckpt 0
python eval.py \
    --data_dir=./coco2017_1000/ \
    --device_target=CPU \
    --pretrained=./coco2017_1000/weight.ckpt