export PYTHONPATH="$(dirname "$(dirname "$0")"):$PYTHONPATH"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export MODEL_DIR="/hpc2hdd/home/zchen379/huggingface_files/stable-diffusion-v1-4"
export TRAIN_DATA_DIR="/hpc2hdd/home/zchen379/working/test123"
export OUTPUT_DIR="outputs/debug/"


accelerate launch --config_file=accelerate_configs/0.yaml --mixed_precision="fp16"  train_controlnet/train_cross_ctrl_uni_pretrain_only_seg.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$TRAIN_DATA_DIR \
 --resolution=256 \
 --learning_rate=1e-5 \
 --train_batch_size=8 \
 --validation_steps=5000 \
 --max_train_steps=300000 \
 --mixed_precision="fp16" \
 --checkpoints_total_limit=10 \
 --resume_from_checkpoint="latest" \
 --miou \
 --test_batch_size=10 \
 --seed=42 \



# Notice: mixed_precision, learning_rate, bs
