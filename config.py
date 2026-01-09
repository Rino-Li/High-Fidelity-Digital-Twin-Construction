import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--g_lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--c_lr", type=float, default=1e-5, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=10, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=32, help="dimensionality of the latent space")
    parser.add_argument("--critic_epoch", type=int, default=2, help="dimensionality of the latent space")
    parser.add_argument("--generate_epoch", type=int, default=200, help="dimensionality of the latent space")
    parser.add_argument("--lambda_gp", type=int, default=5, help="dimensionality of the latent space")
    opt = parser.parse_args()
    return opt