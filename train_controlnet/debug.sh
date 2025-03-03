export PYTHONPATH="$(dirname "$(dirname "$0")"):$PYTHONPATH"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


export MODEL_DIR="/hpc2hdd/home/zchen379/huggingface_files/stable-diffusion-v1-4_x0"
export TRAIN_DATA_DIR="/hpc2hdd/home/zchen379/working/dataset/wenhangge/data/data/objaverse_render_single_env"
export OUTPUT_DIR="debug/"


accelerate launch --config_file=accelerate_configs/8cards.yaml --mixed_precision="fp16" train_controlnet/debug.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$TRAIN_DATA_DIR \
 --resolution=512 \
 --learning_rate=1e-5 \
 --adam_beta1=0.9 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=1 \
 --num_validation_images=5 \
 --validation_steps=50000000 \
 --max_train_steps=3000000 \
 --mixed_precision="fp16" \
 --checkpoints_total_limit=20 \
 --validation_prompt "" "" "" \
 --resume_from_checkpoint="latest" \
 --validation_image "/hpc2hdd/home/zchen379/working/zf_alchemist/diffusion_4_perception/dataset/test_data" \
 --seed=42 \

# Notice: mixed_precision, learning_rate, bs