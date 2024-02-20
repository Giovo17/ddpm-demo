import os
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
from matplotlib import animation

from fire import Fire # Automatically generate CLIs from the main function
from tqdm import tqdm # Progress bar in for loops
from pydantic import BaseModel

from ddpm.ddpm import DDPM
from ddpm.models import BasicDiscreteTimeModel


os.environ["KMP_DUPLICATE_LIB_OK"] = "True" # not sure if this is necessary


# Set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print("Using NVIDIA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


# Set seeds
seed = 42
def set_seed(seed: int = seed) -> None:
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # Set cuda seeds
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set as {seed}")

set_seed()



class TrainResult(BaseModel):
    losses: List[int]
    samples: List[Any]


def train(model: nn.Module, ddpm: DDPM, batch_size: int = 128, n_epochs: int = 400, sample_size: int = 512,
          steps_between_sampling: int = 20, device: str = "cpu") -> TrainResult:

    assert batch_size > 0 and steps_between_sampling > 0 and sample_size > 0

    N = 1 << 10 # Bitwise left shifting: N = 1*2^10
    X = make_swiss_roll(n_samples=N, noise=1e-1)[0][:, [0, 2]] / 10.0

    optim = Adam(model.parameters(), 1e-3)

    losses: List[float] = []
    samples: List[Any] = []
    step = 0
    avg_loss = None  # exponential moving average
    
    with tqdm(total=n_epochs * (len(X) // batch_size)) as pbar:
        for _ in range(n_epochs):
            ids = np.random.choice(N, N, replace=False)
            for i in range(0, len(ids), batch_size):
                x = torch.tensor(X[ids[i : i + batch_size]], dtype=torch.float32).to(device)
                optim.zero_grad()
                loss = ddpm.diffusion_loss(model, x)
                loss.backward()
                optim.step()

                pbar.update(1)
                losses.append(loss.item())
                if avg_loss is None:
                    avg_loss = losses[-1]
                else:
                    avg_loss = 0.95 * avg_loss + 0.05 * losses[-1]
                if not step % 10:
                    pbar.set_description(f"Iter: {step}. Average Loss: {avg_loss:.04f}")
                if not step % steps_between_sampling:
                    samples.append(ddpm.sample(model, n_samples=sample_size))
                step += 1
    
    return TrainResult(losses=losses, samples=samples)


def animate(samples: List[Any], save: bool = True, path: str = ""):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0))
    scat = ax.scatter(*samples[0].detach().numpy().T, c="k", alpha=0.3)

    def animate(i):
        scat.set_offsets(samples[i].detach().numpy())

    anim = animation.FuncAnimation(fig, animate, interval=100, frames=len(samples) - 1)
    
    if save:
        print(f"Animation path: {path}")
        anim.save(path, writer=animation.PillowWriter(fps=5))
    
    return anim


def main(n_steps: int = 100, d_model: int = 128, n_layers: int = 2, batch_size: int = 128, n_epochs: int = 400,
         sample_size: int = 512, steps_between_sampling: int = 20):

    print("Creating model ...")
    model = BasicDiscreteTimeModel(d_model=d_model, n_layers=n_layers).to(device)
    ddpm = DDPM(n_steps=n_steps).to(device).to(device)
    print("Model created!")

    print("Training ...")
    result = train(
        model=model,
        ddpm=ddpm,
        batch_size=batch_size,
        n_epochs=n_epochs,
        sample_size=sample_size,
        steps_between_sampling=steps_between_sampling,
        device=device
    )
    print("Training concluded!")

    path = os.path.join(os.getcwd(), "outputs", "animation.gif")
    animate(result.samples, path)


if __name__ == "__main__":
    Fire(main)
