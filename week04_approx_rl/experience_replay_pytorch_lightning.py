# The aim of this template is to use PyTorch Lightning for Experience Replay.
# Only necessary classes have been provided. Feel free to experiment.

import argparse
import random
from typing import Tuple, List
from collections import namedtuple, deque, OrderedDict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

import pytorch_lightning as pl


Experience = namedtuple('Experience', field_names=['states', 'actions', 'rewards', 'dones', 'next_states'])

class ReplayBuffer:
    """
    The goal of buffer_replay is to store previous experiences.
    """

    def __init__(self, buffer_size: int) -> None:
        """
        Args:
            buffer_size: max number of transitions to store in the buffer.
            When the max is reached, old experiences shall be dropped.
        """
        self.buffer = deque(maxlen = buffer_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Adding more experience to be stored in buffer.
        Args:
            experience: tuple containing: states, actions, rewards, dones, next_states.
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        """
        Sampling experiences -without replacement- from buffer, according to given batch_size.
        This shall later be used with PLDataset class.
        Args:
            batch_size: chosen batch size
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (
            np.array(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states),
        )


class PLDataset(IterableDataset):
    """
    Iterable dataset storing experiences through ReplayBuffer.
    """

    def __init__(self, replay_buffer: ReplayBuffer, batch_size: int = 100) -> None:
        """
        Args:
            replay_buffer: ReplayBuffer containing previous experiences.
            batch_size: size of batch sampled at a time.
        """
        self.buffer_replay = replay_buffer
        self.batch_size = batch_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, next_states = self.buffer_replay.sample(self.batch_size)
        for idx in range(len(dones)):
            yield states[idx], actions[idx], rewards[idx], dones[idx], next_states[idx]


class Agent:
    """
    Agent interacting in the environment.
    """

    def __init__(self, env: str, replay_buffer: ReplayBuffer) -> None:
        """
        This is just a basic class, feel free to play with it.
        Args:
            env: the environment where the agent interact
            replay_buffer: ReplayBuffer where past experiences are stored
        """
        self.env = env
        self.replay_buffer = replay_buffer

    def reset(self) -> None:
        pass

    def get_action(self, net: pl.Lightning, epsilon: float = 0.0, device: str = 'cpu') -> int:
        pass

    @torch.no_grad()
    def play_step(self, net: pl.LightningModule, epsilon: float = 0.0, device: str = 'cpu') -> Tuple[float, bool]:
        pass

class LightningAgent(pl.LightningModule):
    """
    Using PyTorch Lightning
    """

    def __init__(self,
                 env: str,
                 replay_size: int = 100,
                 batch_size: int = 20,
                 lr: float = 1e-2,
                 **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.env = gym.make(env)
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.lr = lr

        self.buffer = ReplayBuffer(self.batch_size)
        self.agent = Agent(self.env, self.buffer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def loss(self, batch: Tuple):
        pass

    def training_step(self, batch: Tuple, batch_idx: int) -> OrderedDict:
        pass

    def configure_optimizers(self):
        pass

    def __dataLoader(self) -> DataLoader:
        """
        Initializing Replay Buffer to be used
        """
        dataset = PLDataset(self.buffer, self.batch_size)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """
        Training
        """
        return self.__dataLoader()


def main(args) ->None:
    model = LightningAgent(**vars(args))

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=1,
    )

    trainer.fit(model)

if __name__ == '__main__':
    main(args)