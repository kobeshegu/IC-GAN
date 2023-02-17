from templates import *

if __name__ == '__main__':
    # train the latent DPM
    # NOTE: only need a single gpu
    gpus = [0]
    conf = run_latent_ddim()
    train(conf, gpus=gpus)

    # gpus = [0, 1]
    # conf.eval_programs = ['fid(10,10)']
    # train(conf, gpus=gpus, mode='eval')