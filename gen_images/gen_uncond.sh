####### generate unconditional images
srun -p digitalcity -N1 --quotatype=reserved --gres=gpu:1 --job-name=icgan128 --cpus-per-task=16 \
python inference/generate_images.py --root_path /mnt/petrelfs/yangmengping/ckpt/ICGAN --model icgan --model_backbone biggan --resolution 256 --save_path /mnt/petrelfs/yangmengping/generate_data/ImageNet256/ICGAN_uncond
