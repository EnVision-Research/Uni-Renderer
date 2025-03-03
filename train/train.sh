export PYTHONPATH="$(dirname "$(dirname "$0")"):$PYTHONPATH"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


export MODEL_DIR="/hpc2hdd/home/zchen379/huggingface_files/stable-diffusion-v1-4_x0"
export OUTPUT_DIR="test_git"
module load cuda/12.1 compilers/gcc-11.1.0 compilers/icc-2023.1.0 cmake/3.27.0
export CXX=$(which g++)
export CC=$(which gcc)
export CPLUS_INCLUDE_PATH=/hpc2ssd/softwares/cuda/cuda-12.1/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
export CUDA_LAUNCH_BLOCKING=1


accelerate launch --config_file=accelerate_configs/01.yaml --main_process_port="14126" --mixed_precision="fp16" train/train.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --resolution=512 \
 --learning_rate=5e-6 \
 --adam_beta1=0.9 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=1 \
 --num_validation_images=9 \
 --validation_steps=5000 \
 --checkpointing_steps=5000 \
 --max_train_steps=5000000 \
 --mixed_precision="fp16" \
 --checkpoints_total_limit=50 \
 --validation_prompt "" "" "" \
 --resume_from_checkpoint="latest" \
 --validation_image "" \
 --seed=97