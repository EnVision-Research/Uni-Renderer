export PYTHONPATH="$(dirname "$(dirname "$0")"):$PYTHONPATH"

export MODEL_DIR="/hpc2hdd/home/zchen379/huggingface_files/stable-diffusion-v1-4"
export TRAIN_DATA_DIR="/hpc2hdd/home/zchen379/dataset/NYUv2_dataset"
export OUTPUT_DIR="outputs/color_rgb_mask/"


accelerate launch --config_file=accelerate_configs/01.yaml --mixed_precision="fp16" train_controlnet/train_cross_ctrl_joint_sample.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$TRAIN_DATA_DIR \
 --resolution=256 \
 --learning_rate=1e-5 \
 --train_batch_size=6 \
 --validation_steps=5000 \
 --max_train_steps=300000 \
 --mixed_precision="fp16" \
 --checkpoints_total_limit=10 \
 --validation_prompt "" "" "" \
 --resume_from_checkpoint="latest"
#  --checkpointing_steps=1 \
#  --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \


# Notice: mixed_precision, learning_rate, bs