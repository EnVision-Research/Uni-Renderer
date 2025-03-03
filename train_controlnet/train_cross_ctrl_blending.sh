export PYTHONPATH="$(dirname "$(dirname "$0")"):$PYTHONPATH"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


export MODEL_DIR="/hpc2hdd/home/zchen379/huggingface_files/stable-diffusion-v1-4"
export TRAIN_DATA_DIR="/hpc2hdd/home/zchen379/working/all_geometry"
export OUTPUT_DIR="outputs/debug123/"


accelerate launch --config_file=accelerate_configs/01.yaml --mixed_precision="fp16" train_controlnet/train_cross_ctrl_blending.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$TRAIN_DATA_DIR \
 --resolution=256 \
 --learning_rate=1e-5 \
 --train_batch_size=8 \
 --validation_steps=5000 \
 --max_train_steps=150000 \
 --mixed_precision="fp16" \
 --checkpoints_total_limit=10 \
 --validation_prompt "" "" "" \
 --resume_from_checkpoint="latest" \
 --validation_image "/hpc2hdd/home/zchen379/working/all_geometry/american_football_1k_gen/roughness_0.2_metallic_0.8_output/val_000/rgba.png" \
 --seed=97 \



# Notice: mixed_precision, learning_rate, bs