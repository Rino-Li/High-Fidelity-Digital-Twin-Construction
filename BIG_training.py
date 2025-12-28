import torch
from BNN_GAN import BayesianGenerator, Discriminator
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from config import get_arguments
from dataset_construct import get_data_iterable
import pyro
from pyro.infer.autoguide import AutoNormal
from tqdm import tqdm

# We have constructed the training process of the Bayesian Inference Generator by referring to that of GANs.

opt = get_arguments()
cuda = True if torch.cuda.is_available() else False

latent_dim = 16
label_dim = 8
output_dim = 6
batch_size = opt.batch_size
num_epochs = opt.epoch


def compute_gradient_penalty(discriminator, real_samples, fake_samples, labels):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates, labels)
    fake = torch.ones(d_interpolates.size(), requires_grad=False).to(real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def model(z, labels, real_data=None, use_discriminator=False, alpha=1.0, n_mc_samples=5):
    pyro.module("discriminator", discriminator)  
    pyro.module("generator", generator) 
    samples = []
    z_i = torch.randn(z.size(0), z.size(1)).to(z.device) 
    sample = generator(z_i, labels) 
    if use_discriminator:
        preds = discriminator(sample, labels)
        loss = -preds.squeeze(-1)
        with pyro.plate("data", size=z.size(0)):
            pyro.factor("generator_loss", alpha * loss)


def train_discriminator(generator, guide, discriminator, optimizer_D, real_data, labels, latent_dim,
                        n_mc_samples=batch_size):
    discriminator.train()
    optimizer_D.zero_grad()
    fake_samples = generate_sample_set(generator, guide, labels, latent_dim, n_mc_samples)
    real_samples = real_data
    real_pred = discriminator(real_samples, labels)
    fake_pred = discriminator(fake_samples, labels)
    gp = compute_gradient_penalty(discriminator, real_samples, fake_samples, labels)
    lambda_gp = 10  
    d_loss = -(torch.mean(real_pred) - torch.mean(fake_pred)) + lambda_gp * gp
    d_loss.backward()
    optimizer_D.step()
    return d_loss.item(), torch.mean(real_pred).item(), torch.mean(fake_pred).item(), gp


def generate_sample_set(generator, guide, labels, latent_dim, n_samples):
    batch_size = labels.size(0)
    z = torch.randn(batch_size, latent_dim).to(labels.device)
    with torch.no_grad():
        sample = generator(z, labels)
    return sample

device = torch.device("cpu")
pyro.clear_param_store()
generator = BayesianGenerator(latent_dim, label_dim, output_dim).to(device)
discriminator = Discriminator(output_dim=output_dim, n_samples=batch_size, label_dim=label_dim)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-5)
guide = AutoNormal(model)
optimizer_G = pyro.optim.Adam({"lr": 1e-4})
elbo = Trace_ELBO()
svi = SVI(model, guide, optimizer_G, loss=elbo)
dataloader = get_data_iterable()
warmup_epochs = 5
mc_samples = 3
for epoch in tqdm(range(num_epochs)):
    for i, (real_data, labels) in enumerate(dataloader):
        batch_data = real_data.cuda()
        batch_size = batch_data.size(0)
        for _ in range(opt.critic_epoch):
            d_loss, real_score, fake_score, gp = train_discriminator(generator, guide, discriminator, optimizer_D, data,
                                                                     labels, latent_dim,
                                                                     n_mc_samples=batch_size)
        z = torch.randn(data.size(0), latent_dim).to(device)
        if epoch < warmup_epochs:
            svi.step(z, labels, real_data=data, use_discriminator=False, n_mc_samples=batch_size)
            avg_elbo = 0
        else:
            total_elbo = 0
            for _ in range(mc_samples):
                total_elbo += svi.step(z, labels, real_data=data, use_discriminator=True, alpha=0.1,
                                       n_mc_samples=batch_size)
            avg_elbo = total_elbo / mc_samples
        print(
            f"[Epoch {epoch + 1}/{num_epochs}] D_loss={d_loss:.4f}, ELBO_G={'{:.4f}'.format(avg_elbo) if epoch >= warmup_epochs else 'warm-up'}")
        print("real_score:", real_score, "  fake_score:", fake_score, "  gp", gp)
torch.save({
    "generator_state_dict": generator.state_dict(),
    "guide_param_store": pyro.get_param_store().get_state()  
}, "generator_bayesian.pth")
print("Generator model saved to final_generator.pth")

generator.eval()

with torch.no_grad():
    for i, (real_data,) in enumerate(dataloader):  
        if i > 0:
            break
        batch_data = real_data.cuda()
        batch_size = batch_data.size(0)

        data = batch_data[:, 2:].to(device)
        print('-------real data----------')
        print(data)

        labels = batch_data.to(device)
        z = torch.randn(batch_size, latent_dim).to(device)
        gen_samples = generator(z, labels)
        print('-------gen data----------')
        print("Generated Samples:", gen_samples)
