export PYTHONPATH="$(dirname "$(dirname "$0")"):$PYTHONPATH"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


export MODEL_DIR="/hpc2hdd/home/zchen379/huggingface_files/stable-diffusion-v1-4_custom"
export TRAIN_DATA_DIR="/hpc2hdd/home/zchen379/working/new_dataset_0409/dataset/wenhangge/data/data/inver_data"
export OUTPUT_DIR="outputs-0409/debug123_3mod_vpred_512_large_1e-4/"


accelerate launch --config_file=accelerate_configs/0123.yaml --mixed_precision="fp16" train_controlnet/train_cross_ctrl_blending_3mod_vpred.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$TRAIN_DATA_DIR \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=2 \
 --validation_steps=2000 \
 --max_train_steps=300000 \
 --mixed_precision="fp16" \
 --checkpoints_total_limit=10 \
 --validation_prompt "" "" "" \
 --resume_from_checkpoint="latest" \
 --validation_image "/hpc2hdd/home/zchen379/working/zf_alchemist/diffusion_4_perception/dataset/test_data_new" \
 --seed=42 \

# Notice: mixed_precision, learning_rate, bs