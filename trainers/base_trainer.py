"""
Common PyTorch trainer code.
"""

# System
import os
import re
import logging
import time

# Externals
import numpy as np
import pandas as pd
import torch

class BaseTrainer(object):
    """
    Base class for PyTorch trainers.
    This implements the common training logic,
    logging of summaries, and checkpoints.
    """

    def __init__(self, output_dir=None, gpu=None, distributed=False, rank=0):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = (os.path.expandvars(output_dir)
                           if output_dir is not None else None)
        self.gpu = gpu
        if gpu is not None:
            self.device = 'cuda:%i' % gpu
        else:
            self.device = 'cpu'
        self.distributed = distributed
        self.summaries = None
        self.summary_file = None
        self.rank = rank

    def print_model_summary(self):
        """Override as needed"""
        self.logger.info(
            'Model: \n%s\nParameters: %i' %
            (self.model, sum(p.numel()
             for p in self.model.parameters()))
        )

    def save_summary(self, summaries):
        """Save summary information.
        This implementation is currently inefficient for simplicity:
            - we build a new DataFrame each time
            - we write the whole summary file each time
        """
        if self.summaries is None:
            self.summaries = pd.DataFrame([summaries])
        else:
            self.summaries = self.summaries.append([summaries], ignore_index=True)
        if self.output_dir is not None:
            summary_file = os.path.join(self.output_dir, 'summaries.csv')
            self.summaries.to_csv(summary_file, index=False)

    def write_summaries(self):
        """Deprecated"""
        assert self.output_dir is not None
        summary_file = os.path.join(self.output_dir, 'summaries.npz')
        self.logger.info('Saving summaries to %s' % summary_file)
        np.savez(summary_file, **self.summaries)

    def write_checkpoint(self, checkpoint_id, **kwargs):
        """Write a checkpoint for the model"""
        assert self.output_dir is not None
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = 'model_checkpoint_%03i.pth.tar' % checkpoint_id
        torch.save(kwargs, os.path.join(checkpoint_dir, checkpoint_file))

    def load_checkpoint(self, checkpoint_id=-1):
        """Load a model checkpoint"""
        assert self.output_dir is not None
        # Load the summaries
        summary_file = os.path.join(self.output_dir, 'summaries.csv')
        logging.info('Reloading summary at %s', summary_file)
        self.summaries = pd.read_csv(summary_file)
        # Load the checkpoint
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        if checkpoint_id == -1:
            # Find the last checkpoint
            last_checkpoint = sorted(os.listdir(checkpoint_dir))[-1]
            pattern = 'model_checkpoint_(\d..).pth.tar'
            checkpoint_id = int(re.match(pattern, last_checkpoint).group(1))
        checkpoint_file = 'model_checkpoint_%03i.pth.tar' % checkpoint_id
        logging.info('Reloading checkpoint at %s', checkpoint_file)
        return torch.load(os.path.join(checkpoint_dir, checkpoint_file))

    def build_model(self):
        """Virtual method to construct the model(s)"""
        raise NotImplementedError

    def train_epoch(self, data_loader):
        """Virtual method to train a model"""
        raise NotImplementedError

    def evaluate(self, data_loader):
        """Virtual method to evaluate a model"""
        raise NotImplementedError

    def train(self, train_data_loader, n_epochs, valid_data_loader=None):
        """Run the model training"""

        # Loop over epochs
        start_epoch = 0
        if self.summaries is not None:
            start_epoch = self.summaries.epoch.max() + 1
        for i in range(start_epoch, n_epochs):
            self.logger.info('Epoch %i' % i)
            summary = dict(epoch=i)
            # Train on this epoch
            start_time = time.time()
            summary.update(self.train_epoch(train_data_loader))
            summary['train_time'] = time.time() - start_time
            # Evaluate on this epoch
            if valid_data_loader is not None:
                start_time = time.time()
                summary.update(self.evaluate(valid_data_loader))
                summary['valid_time'] = time.time() - start_time
            # Save summary, checkpoint
            self.save_summary(summary)
            if self.output_dir is not None:
                self.write_checkpoint(checkpoint_id=i)

        return self.summaries
