cd /mnt/nas_3dv/hdd1/fangchuan/EscherNet/6DoF

WANDB_MODE=offline accelerate launch --config_file ../configs/accelerate-train.yaml \
            train_eschernet_koolai.py --train_data_dir /mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/ \
            --train_split_file /mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/train.txt \
            --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
            --train_batch_size 1 \
            --dataloader_num_workers 4 \
            --mixed_precision bf16 \
            --gradient_checkpointing \
            --T_in 1 \
            --T_out 1 \
            --T_in_val 3 \
            --output_dir logs_pano_eschernet_6dof \
            --checkpoints_total_limit 10 
            # --push_to_hub --hub_model_id ***** --hub_token hf_******************* --tracker_project_name eschernet