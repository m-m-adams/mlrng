# %%
import math
import numpy as np
import torch
import torch.nn as nn
from gan_models import Generator, Discriminator
from gendata import generate_even_data
from IPython.display import clear_output
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline


def train(max_int: int = 128, batch_size: int = 10_000, training_steps: int = 500):

    # Optimizers

    # loss
    loss = nn.BCELoss()

    for i in range(training_steps):
        # zero the gradients on each iteration
        generator_optimizer.zero_grad()

        # Create noisy input for generator
        # Need float type instead of int
        seeds = torch.randint(0, 2, size=(batch_size, input_length*4)).float()
        generated_data = generator(seeds)

        # Train the generator
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true.
        discriminator_out = discriminator(seeds)
        generator_loss = loss(torch.round(generated_data),
                              torch.round(1-discriminator_out.detach()))
        generator_loss.backward()
        generator_optimizer.step()

        # add .detach() here think about this
        discriminator_loss = loss(discriminator_out, generated_data.detach())
        discriminator_loss.backward()
        discriminator_optimizer.step()

        print(generator_loss.detach(), discriminator_loss.detach(), i)
        if (i+1) % 100 == 0:
            live_plot(prepare_output(generated_data, discriminator_out))


def prepare_output(generated, discriminated):
    out = []
    for g, d in zip(generated, discriminated):
        g = torch.round(g).detach().numpy()
        d = torch.round(d).detach().numpy()
        g_v = sum(x*(2**i) for i, x in enumerate(g))
        d_v = sum(x*(2**i) for i, x in enumerate(d))
        out.append([g_v, d_v])
    return out

# %%


def live_plot(x, figsize=(7, 5), title=''):
    x = np.array(x)[:, 0]
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    plt.hist(x, label='distribution', color='k', bins=128, range=(0, 128))

    plt.title(title)
    plt.grid(True)
    plt.xlabel('output number')
    plt.ylabel('count')
    plt.show()


input_length = int(math.log(128, 2))
generator = Generator(input_length)
discriminator = Discriminator(input_length)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0005)
discriminator_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=0.0005)

train()

# %%
n2gen = 10_000
seeds = torch.randint(0, 2, size=(input_length*4,)).float()

for _ in range(n2gen):
    next = torch.round(generator(seeds)).detach()
    seeds = torch.cat((seeds[input_length:], next), 0)
    n = sum(x*(2**i) for i, x in enumerate(next.detach().numpy()))
    with open("rngout.txt", 'a') as f:
        f.write(f"{int(n)}\n")
# %%
