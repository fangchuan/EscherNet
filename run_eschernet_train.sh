cd /mnt/nas_3dv/hdd1/fangchuan/EscherNet/6DoF
accelerate launch train_eschernet_koolai.py --train_data_dir /mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/ \
            --train_split_file /mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/train.txt \
            --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
            --train_batch_size 256 \
            --dataloader_num_workers 16 \
            --mixed_precision bf16 \
            --gradient_checkpointing \
            --T_in 3 \
            --T_out 3 \
            --T_in_val 7 \
            --output_dir pano-eschernet-6dof \
            --checkpoints_total_limit 10 
            # --push_to_hub --hub_model_id ***** --hub_token hf_******************* --tracker_project_name eschernet