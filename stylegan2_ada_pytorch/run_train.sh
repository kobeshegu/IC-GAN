srun -p digitalcity -N1 --quotatype=reserved --job-name=ICGAN --gres=gpu:1 --cpus-per-task=16 \
        python -m pdb run.py \
        --json_config config_files/COCO_Stuff/IC-GAN/icgan_stylegan_res128.json \
        --data_root /mnt/petrelfs/yangmengping/ICGAN/coco \
        --base_root /mnt/petrelfs/yangmengping/ICGAN \
        --slurm True \
