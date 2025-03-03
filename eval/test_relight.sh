export PYTHONPATH="$(dirname "$(dirname "$0")"):$PYTHONPATH"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export PRETRAIN_MODEL_DIR="/hpc2hdd/home/zchen379/huggingface_files/stable-diffusion-v1-4_x0"
export MODEL_DIR="/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/14w_highmel_high_rough/checkpoint-340000"
export OUTPUT_DIR='/hpc2hdd/home/zchen379/sd3/diff_rendering_unet/dataset/other_intrinsics'
module load cuda/12.1 compilers/gcc-11.1.0 compilers/icc-2023.1.0 cmake/3.27.0
export CXX=$(which g++)
export CC=$(which gcc)
export CPLUS_INCLUDE_PATH=/hpc2ssd/softwares/cuda/cuda-12.1/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH

accelerate launch --config_file=accelerate_configs/0.yaml --mixed_precision="fp16" eval/test_relight.py \
 --pretrained_model_name_or_path=$PRETRAIN_MODEL_DIR \
 --controlnet_model_name_or_path=$MODEL_DIR \
 --num_validation_images=8 \
 --output_dir=$OUTPUT_DIR \
 --resolution=512 \
 --step="592000" \
 --mixed_precision="fp16" \
 --validation_prompt "" "" "" \
 --validation_image "/hpc2hdd/home/zchen379/working/zf_alchemist/diffusion_4_perception/dataset/test_data_more_more" \
 --seed=42 \

# Notice: mixed_precision, learning_rate, bs 97, 1024， 22, 1010(orange) 4621
