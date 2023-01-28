####### generate unconditional images
srun -p digitalcity -N1 --quotatype=reserved --gres=gpu:1 --job-name=icgan --cpus-per-task=16 \
python inference/generate_images.py --root_path /mnt/petrelfs/yangmengping/ckpt/ICGAN --model icgan --model_backbone biggan --resolution 128 --save_path /mnt/petrelfs/yangmengping/generate_data/ImageNet128/ICGAN_uncond
