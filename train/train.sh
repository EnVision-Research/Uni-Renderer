export PYTHONPATH="$(dirname "$(dirname "$0")"):$PYTHONPATH"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


# files and directories
export MODEL_DIR="/hpc2hdd/home/zchen379/huggingface_files/stable-diffusion-v1-4_x0"
export OUTPUT_DIR="test_git"
export DATASET_ROOT_DIR='/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/Objaverse_OBJ_Mesh_valid'
export DATASET_ENV_DIR='/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/env_mipmap_gaint'
export DATASET_FILTER_OBJ_JSON='/hpc2hdd/home/zchen379/sd3/objaverse_data/Mesh_final_valid_texturemap.json' 
export DATASET_TEST_OBJ_JSON='/hpc2hdd/home/zchen379/sd3/objaverse_data/test_ood_one.json'

# dataset generation
module load cuda/12.1 compilers/gcc-11.1.0 compilers/icc-2023.1.0 cmake/3.27.0
export CXX=$(which g++)
export CC=$(which gcc)
export CPLUS_INCLUDE_PATH=/hpc2ssd/softwares/cuda/cuda-12.1/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
export CUDA_LAUNCH_BLOCKING=1


accelerate launch --config_file=accelerate_configs/01.yaml --main_process_port="14126" --mixed_precision="fp16" train/train.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_root_dir=$DATASET_ROOT_DIR \
 --dataset_env_dir=$DATASET_ENV_DIR \
 --dataset_filter_obj_json=$DATASET_FILTER_OBJ_JSON \
 --dataset_test_obj_json=$DATASET_TEST_OBJ_JSON \
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