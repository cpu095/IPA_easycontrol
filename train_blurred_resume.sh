export MODEL_DIR="/opt/liblibai-models/model-weights/black-forest-labs/FLUX.1-dev" # your flux path
export OUTPUT_DIR="/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/IPA_easycontrol/models/blurred_model_0411"  # your save path
export CONFIG="./default_config.yaml"
export TRAIN_DATA="/opt/liblibai-models/user-workspace/zhangyuxuan/project/gc_project/projects2/data_process/unicontrol/canny_all.jsonl"
export LOG_PATH="$OUTPUT_DIR/log"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $CONFIG train_resume.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --pretrained_lora_path "/opt/liblibai-models/user-workspace/songyiren/FYP/sjc/redux/models/blurred_model_0408/checkpoint-1000/lora.safetensors" \
    --cond_size=512 \
    --noise_size=1024 \
    --subject_column="None" \
    --spacial_column="control_canny" \
    --target_column="source" \
    --caption_column="prompt" \
    --ranks 128 \
    --network_alphas 128 \
    --lora_num 1 \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --validation_prompt "A bird in a library" \
    --num_train_epochs=1000 \
    --validation_steps=1000 \
    --checkpointing_steps=1000 \
    --subject_test_images None \
    --spatial_test_images "/opt/liblibai-models/user-workspace/zhangyuxuan/project/easycontrol/code0221/test_imgs/canny.jpg" \
    --test_h 1024 \
    --test_w 1024 \
    --num_validation_images=2
