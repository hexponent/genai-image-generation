import os
from pathlib import Path
import re

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import tqdm


class TrainableModule(nn.Module):
    device = 'cpu'  # default device
    weights_dir_name = 'weights'
    tensorboard_dir_name = 'runs'

    def __init__(self, name, resources_dir=None):
        super().__init__()

        self.name = name

        self.resources_dir = resources_dir or os.path.abspath("")

    @property
    def resources_dir(self):
        return self._resources_dir

    @resources_dir.setter
    def resources_dir(self, value):
        if not os.path.isabs(value):
            raise ValueError('resources_dir should be an absolute path')
        self._resources_dir = Path(value)
        print(f"{self.__class__.__name__}<{self.name}> now uses {self._resources_dir} for resources.")

    @property
    def weights_dir(self):
        return self.resources_dir / self.weights_dir_name

    @property
    def tensorboard_dir(self):
        return self.resources_dir / self.tensorboard_dir_name

    def to(self, *args, **kwargs):
        super_res = super().to(*args, **kwargs)

        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device:
            self.device = device.type

        return super_res

    def _train_setup(self):
        """Training setup hook"""

    def _train_teardown(self):
        """Training teardown hook"""

    def _train_epoch(self):
        raise NotImplementedError

    def train_loop(self, data, epochs=100, **extra_train_data):
        epoch_start = self.load_pretrained()

        self.train_data, self.val_data = data
        self.extra_train_data = extra_train_data

        self.writer = SummaryWriter(self.tensorboard_dir / self.name)
        weights_path = self.weights_dir / self.name
        weights_path.mkdir(parents=True, exist_ok=True)

        self._train_setup()

        for epoch_n in range(epoch_start, epochs + 1):
            print(f"Epoch {epoch_n:>{len(str(epochs))}}/{epochs}")

            losses = self._train_epoch()
            for loss_type, loss_value in losses.items():
                print(f"{loss_type.capitalize()} loss: {loss_value}")
                self.writer.add_scalar(f"Loss/{loss_type}", loss_value, epoch_n)

            epoch_val_loss = list(losses.values())[-1]
            if epoch_val_loss < self.best_loss:
                self.best_loss = epoch_val_loss
                torch.save(self.state_dict(), weights_path / 'best.pth')

            if epoch_n % 5 == 0:
                torch.save(self.state_dict(), weights_path / f'{epoch_n}.pth')

                self._epoch_sample_visualization(epoch_n)

            self.writer.flush()

        self.writer.close()
        self._train_teardown()

    def load_pretrained(self):
        files = list(map(str, self.weights_dir.rglob(f'{self.name}/*.pth')))

        saved_epochs = {}
        for weights_path in files:
            if epoch := re.search(rf'/{self.name}/(\d+).pth$', weights_path):
                saved_epochs[int(epoch.group(1))] = weights_path

        if saved_epochs:
            latest_epoch = max(saved_epochs)
            latest_weights = saved_epochs[latest_epoch]

            self.load_state_dict(torch.load(latest_weights))
            epoch_start = latest_epoch+1
        else:
            epoch_start = 1
        
        return epoch_start


class TrainValidationTrainableModule(TrainableModule):

    def _run_batches(self, batches, optimizer=None):
        if optimizer:
            self.train()  # Activate training mode
        else:
            self.eval()  # Activate inference mode

        epoch_losses = []
        for batch_data, batch_label in tqdm.tqdm(batches):
            batch_data = batch_data.to(self.device)
            output_batch_data = self(batch_data)

            batch_loss = self.loss(batch_data, output_batch_data)
            epoch_losses.append(batch_loss.item())

            if optimizer:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

        return sum(epoch_losses) / len(epoch_losses)

    def _train_setup(self):
        self.best_loss = float('inf')
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-5, weight_decay=1e-8
        )

        if 'sample_image' not in self.extra_train_data:
            raise ValueError('sample_image is required for encoder/decoder training')

        self.sample_image = self.extra_train_data['sample_image']

        self.writer.add_image('epoch_sample', self.sample_image, global_step=0)

    def _train_epoch(self):
        return {
            'train': self._run_batches(self.train_data, self.optimizer),
            'validation': self._run_batches(self.val_data)
        }

    def _epoch_sample_visualization(self, epoch_n):
        res = self(torch.unsqueeze(self.sample_image, 0).to(self.device))
        if isinstance(res, (list, tuple)):
            res = res[0]
        predict = res.squeeze().detach().to('cpu')
        # save image example to tensorboard
        self.writer.add_image('epoch_sample', predict, global_step=epoch_n)
