export PYTHONPATH="$(dirname "$(dirname "$0")"):$PYTHONPATH"

export MODEL_DIR="/hpc2hdd/home/zchen379/huggingface_files/stable-diffusion-v1-4"
export TRAIN_DATA_DIR="/hpc2hdd/home/zchen379/dataset/NYUv2_dataset"
export OUTPUT_DIR="outputs/sd_condition_exp2/"


accelerate launch --config_file=accelerate_configs/01.yaml --mixed_precision="fp16" train_controlnet/train_cross_ctrl_uni.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$TRAIN_DATA_DIR \
 --resolution=256 \
 --learning_rate=1e-5 \
 --train_batch_size=6 \
 --validation_steps=1 \
 --max_train_steps=300000 \
 --mixed_precision="fp16" \
 --checkpoints_total_limit=10 \
 --validation_prompt "" "" "" \
 --resume_from_checkpoint="latest" \
 --validation_image "/hpc2hdd/home/zchen379/working/new_d4p/conditioning_image_0.png" \
 "/hpc2hdd/home/zchen379/working/new_d4p/conditioning_image_1.png" \
 "/hpc2hdd/home/zchen379/working/new_d4p/conditioning_image_2.png" \
 "/hpc2hdd/home/zchen379/working/new_d4p/conditioning_image_3.png" \
 "/hpc2hdd/home/zchen379/working/new_d4p/conditioning_image_4.png"



# Notice: mixed_precision, learning_rate, bs