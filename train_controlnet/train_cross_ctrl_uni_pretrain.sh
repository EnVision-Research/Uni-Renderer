export PYTHONPATH="$(dirname "$(dirname "$0")"):$PYTHONPATH"

export MODEL_DIR="/hpc2hdd/home/zchen379/huggingface_files/stable-diffusion-v1-4"
export TRAIN_DATA_DIR="/hpc2hdd/home/zchen379/dataset/NYUv2_dataset"
export OUTPUT_DIR="outputs/sd_condition_w_pretrain_exp3_inferstep_50_guidance_2.5_w_metrics"


accelerate launch --config_file=accelerate_configs/01.yaml --mixed_precision="fp16" train_controlnet/train_cross_ctrl_uni_pretrain.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$TRAIN_DATA_DIR \
 --resolution=256 \
 --learning_rate=1e-5 \
 --train_batch_size=6 \
 --validation_steps=5000 \
 --max_train_steps=150000 \
 --mixed_precision="fp16" \
 --checkpoints_total_limit=10 \
 --validation_prompt "" "" "" \
 --resume_from_checkpoint="latest" \
 --validation_image "/hpc2hdd/home/zchen379/working/new_d4p/conditioning_image_1.png" \
  --miou \
 --test_batch_size=10 \
 --seed=42 \



# Notice: mixed_precision, learning_rate, bs