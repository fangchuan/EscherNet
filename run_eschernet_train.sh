cd ./6DoF

SD_15_PRETEAINED_FOLDER=/mnt/nas_3dv/hdd1/datasets/fangchuan/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9/
CONVNEXT_PRETRRAINED_FOLDER=/mnt/nas_3dv/hdd1/datasets/fangchuan/.cache/huggingface/hub/models--facebook--convnextv2-tiny-22k-224/snapshots/98c27c19bb5d32a35ecdb84bc2050b5ad99686c8/
ALEXNET_PRETRAINED_MODEL_PATH=/mnt/nas_3dv/hdd1/datasets/fangchuan/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth

# CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline accelerate launch --config_file ../configs/accelerate-train.yaml \
accelerate launch --config_file ../configs/accelerate-train.yaml \
            train_eschernet_koolai.py --train_data_dir /mnt/nas_3dv/hdd1/datasets/datasets/KoolAI/processed_data_20240413/ \
            --train_split_file /mnt/nas_3dv/hdd1/datasets/datasets/KoolAI/processed_data_20240413/train.txt \
            --pretrained_model_name_or_path $SD_15_PRETEAINED_FOLDER \
            --train_batch_size 20 \
            --dataloader_num_workers 8 \
            --mixed_precision bf16 \
            --gradient_checkpointing \
            --T_in 3 \
            --T_out 3 \
            --T_in_val 3 \
            --output_dir logs_pano_eschernet_6dof_dino \
            --resume_from_checkpoint latest \
            --checkpoints_total_limit 10 \
            --convnext_pretrained_model_path $CONVNEXT_PRETRRAINED_FOLDER \
            --alexnet_pretrained_model_path $ALEXNET_PRETRAINED_MODEL_PATH
            # --push_to_hub --hub_model_id ***** --hub_token hf_******************* --tracker_project_name eschernet