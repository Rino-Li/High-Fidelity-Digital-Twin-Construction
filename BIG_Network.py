import torch.nn as nn
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
import torch
import pyro
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, output_dim, n_samples, label_dim):
        super().__init__()
        self.input_dim = n_samples * (output_dim + label_dim)
        self.model = nn.Sequential(
            spectral_norm(nn.Linear(self.input_dim, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(512, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(512, 1)),
        )

    def forward(self, x_samples, labels):
        x = torch.cat([x_samples, labels], dim=-1)
        x = x.reshape(1, -1)
        return self.model(x)


class BayesianGenerator(PyroModule):
    def __init__(self, latent_dim, label_dim, output_dim):
        super().__init__()
        prior_std = 0.5 
        self.fc1 = PyroModule[nn.Linear](latent_dim + label_dim, 512)
        self.fc1.weight = PyroSample(dist.Normal(0., prior_std).expand([512, latent_dim + label_dim]).to_event(2))
        self.ln1 = nn.LayerNorm(512)

        self.fc2 = PyroModule[nn.Linear](512, 256)
        self.fc2.weight = PyroSample(dist.Normal(0., prior_std).expand([256, 512]).to_event(2))
        self.ln2 = nn.LayerNorm(256)

        self.out = PyroModule[nn.Linear](256, output_dim)
        self.out.weight = PyroSample(
            dist.Normal(0., prior_std).expand([output_dim, 256]).to_event(2))

        self.act = nn.LeakyReLU(0.2)
        self.output_act = nn.Tanh()

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=-1)
        x = self.act(self.ln1(self.fc1(x)))
        x = self.act(self.ln2(self.fc2(x)))
        return self.output_act(self.out(x))

