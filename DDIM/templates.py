from exp_latent_ddim import *


def ddpm():
    """
    base configuration for all DDIM-based models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_ddpm
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf

def latent_base():
    """
    base configuration for all latent ddim models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.latentnet
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 200
    conf.make_model_conf()
    return conf

def autoenc_base():
    """
    base configuration for all Diff-AE models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_autoenc 
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf


def ffhq64_ddpm():
    conf = ddpm()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 72_000_000
    conf.scale_up_gpus(4)
    return conf


def ffhq64_autoenc():
    conf = autoenc_base()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 72_000_000
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.scale_up_gpus(4)
    conf.make_model_conf()
    return conf


def celeba64d2c_ddpm():
    conf = ffhq128_ddpm()
    conf.data_name = 'celebalmdb'
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 72_000_000
    conf.name = 'celeba64d2c_ddpm'
    return conf


def celeba64d2c_autoenc():
    conf = ffhq64_autoenc()
    conf.data_name = 'celebalmdb'
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 72_000_000
    conf.name = 'celeba64d2c_autoenc'
    return conf


def ffhq128_ddpm():
    conf = ddpm()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 48_000_000
    conf.img_size = 128
    conf.net_ch = 128
    # channels:
    # 3 => 128 * 1 => 128 * 1 => 128 * 2 => 128 * 3 => 128 * 4
    # sizes:
    # 128 => 128 => 64 => 32 => 16 => 8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.scale_up_gpus(4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf()
    return conf


def ffhq128_autoenc_base():
    conf = autoenc_base()
    conf.data_name = 'ffhqlmdb256'
    conf.scale_up_gpus(4)
    conf.img_size = 128
    conf.net_ch = 128
    # final resolution = 8x8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    # final resolution = 4x4
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf()
    return conf


def ffhq256_autoenc():
    conf = ffhq128_autoenc_base()
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 64
    conf.make_model_conf()
    conf.name = 'ffhq256_autoenc'
    return conf


def ffhq256_autoenc_eco():
    conf = ffhq128_autoenc_base()
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 64
    conf.make_model_conf()
    conf.name = 'ffhq256_autoenc_eco'
    return conf


def ffhq128_ddpm_72M():
    conf = ffhq128_ddpm()
    conf.total_samples = 72_000_000
    conf.name = 'ffhq128_ddpm_72M'
    return conf


def ffhq128_autoenc_72M():
    conf = ffhq128_autoenc_base()
    conf.total_samples = 72_000_000
    conf.name = 'ffhq128_autoenc_72M'
    return conf


def ffhq128_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq128_ddpm_130M'
    return conf


def ffhq128_autoenc_130M():
    conf = ffhq128_autoenc_base()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq128_autoenc_130M'
    return conf


def horse128_ddpm():
    conf = ffhq128_ddpm()
    conf.data_name = 'horse256'
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'horse128_ddpm'
    return conf


def horse128_autoenc():
    conf = ffhq128_autoenc_base()
    conf.data_name = 'horse256'
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'horse128_autoenc'
    return conf

def coco128_autoenc():
    conf = ffhq128_autoenc_base()
    conf.data_name = 'coco128'
    conf.total_samples = 120_000
    conf.eval_ema_every_samples = 10_000
    conf.eval_every_samples = 10_000
    conf.name = 'coco128_autoenc'
    return conf

def bedroom128_ddpm():
    conf = ffhq128_ddpm()
    conf.data_name = 'bedroom256'
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.total_samples = 120_000_000
    conf.name = 'bedroom128_ddpm'
    return conf


def bedroom128_autoenc():
    conf = ffhq128_autoenc_base()
    conf.data_name = 'bedroom256'
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.total_samples = 120_000_000
    conf.name = 'bedroom128_autoenc'
    return conf


def pretrain_celeba64d2c_72M():
    conf = celeba64d2c_autoenc()
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'checkpoints/{celeba64d2c_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{celeba64d2c_autoenc().name}/latent.pkl'
    return conf


def pretrain_ffhq128_autoenc72M():
    conf = ffhq128_autoenc_base()
    conf.postfix = ''
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'checkpoints/{ffhq128_autoenc_72M().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{ffhq128_autoenc_72M().name}/latent.pkl'
    return conf


def pretrain_ffhq128_autoenc130M():
    conf = ffhq128_autoenc_base()
    conf.pretrain = PretrainConfig(
        name='130M',
        path=f'checkpoints/{ffhq128_autoenc_130M().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{ffhq128_autoenc_130M().name}/latent.pkl'
    return conf


def pretrain_ffhq256_autoenc():
    conf = ffhq256_autoenc()
    conf.pretrain = PretrainConfig(
        name='90M',
        path=f'checkpoints/{ffhq256_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{ffhq256_autoenc().name}/latent.pkl'
    return conf


def pretrain_horse128():
    conf = horse128_autoenc()
    conf.pretrain = PretrainConfig(
        name='82M',
        path=f'checkpoints/{horse128_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{horse128_autoenc().name}/latent.pkl'
    return conf


def pretrain_bedroom128():
    conf = bedroom128_autoenc()
    conf.pretrain = PretrainConfig(
        name='120M',
        path=f'checkpoints/{bedroom128_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{bedroom128_autoenc().name}/latent.pkl'
    return conf

def pretrain_coco128():
    conf = coco128_autoenc()
    conf.pretrain = PretrainConfig(
        name='120M',
        path=f'checkpoints/{coco128_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{coco128_autoenc().name}/latent.pkl'
    return conf



def latent_mlp_2048_norm_10layers(conf: TrainConfig):
    conf.net_latent_net_type = LatentNetType.skip
    conf.net_latent_layers = 10
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    conf.net_latent_activation = Activation.silu
    conf.net_latent_num_hid_channels = 2048
    conf.net_latent_use_norm = True
    conf.net_latent_condition_bias = 1
    return conf


def latent_mlp_2048_norm_20layers(conf: TrainConfig):
    conf = latent_mlp_2048_norm_10layers(conf)
    conf.net_latent_layers = 20
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    return conf

def latent_diffusion_config(conf: TrainConfig):
    conf.batch_size = 128
    conf.train_mode = TrainMode.latent_diffusion
    conf.latent_gen_type = GenerativeType.ddim
    conf.latent_loss_type = LossType.mse
    conf.latent_model_mean_type = ModelMeanType.eps
    conf.latent_model_var_type = ModelVarType.fixed_large
    conf.latent_rescale_timesteps = False
    conf.latent_clip_sample = False
    conf.latent_T_eval = 20
    conf.latent_znormalize = False
    conf.total_samples = 100_000
    conf.sample_every_samples = 20_000
    conf.eval_every_samples = 20_000
    conf.eval_ema_every_samples = 20_000
    conf.save_every_samples = 20_000
    return conf


def latent_diffusion128_config(conf: TrainConfig):
    conf = latent_diffusion_config(conf)
    conf.batch_size_eval = 1024
    return conf

def latent_2048_batch_size(conf: TrainConfig):
    conf.batch_size = 2048
    conf.eval_ema_every_samples = 20_000
    conf.eval_every_samples = 20_000
    conf.sample_every_samples = 20_000
    conf.save_every_samples = 20_000
    conf.total_samples = 80_000
    return conf

def run_latent_ddim():
    conf = latent_base()
    conf.scale_up_gpus(1)
    conf = latent_diffusion128_config(conf)
    conf = latent_2048_batch_size(conf)
    conf = latent_mlp_2048_norm_20layers(conf)
    conf.total_samples = 100_000_000
    conf.latent_beta_scheduler = 'const0.008'
    conf.latent_loss_type = LossType.l1
    conf.name = 'latent_ddim'
    conf.data_name = 'latent_feature'
    return conf
