cd ./6DoF

# SD_15_PRETEAINED_FOLDER=/mnt/nas_3dv/hdd1/datasets/fangchuan/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9/
SD_15_PRETEAINED_FOLDER=/seaweedfs/training/experiments/zhenqing/cache/models--runwayml--stable-diffusion-v1-5
CONVNEXT_PRETRRAINED_FOLDER=/seaweedfs/training/experiments/zhenqing/cache/facebook-convnextv2-tiny-22k-224
ALEXNET_PRETRAINED_MODEL_PATH=/seaweedfs/training/experiments/zhenqing/cache/lpips/alex.pth

export WANDB_API_KEY=126fd5b8d8ea21e7c9d3dfd0078c8c7ff64187d0
# CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline accelerate launch --config_file ../configs/accelerate-train.yaml \
accelerate launch --config_file ../configs/accelerate-train.yaml \
            train_eschernet_koolai.py --train_data_dir /seaweedfs/training/dataset/qunhe/PanoRoom/processed_data/ \
            --train_split_file /seaweedfs/training/dataset/qunhe/PanoRoom/processed_data/train.txt \
            --pretrained_model_name_or_path $SD_15_PRETEAINED_FOLDER \
            --train_batch_size 4 \
            --dataloader_num_workers 8 \
            --mixed_precision bf16 \
            --gradient_checkpointing \
            --T_in 3 \
            --T_out 3 \
            --T_in_val 3 \
            --output_dir logs_pano_eschernet_6dof_dino_qunheeeds \
            --resume_from_checkpoint latest \
            --checkpoints_total_limit 10 \
            --convnext_pretrained_model_path $CONVNEXT_PRETRRAINED_FOLDER \
            --alexnet_pretrained_model_path $ALEXNET_PRETRAINED_MODEL_PATH
            # --push_to_hub --hub_model_id ***** --hub_token hf_******************* --tracker_project_name eschernet