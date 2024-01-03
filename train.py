import os
from datetime import datetime

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate
from onsets_and_frames import *
import onsets_and_frames.transcriber as nnmodels

from onsets_and_frames.dataset import collating_function

ex = Experiment('train_transcriber')


class EarlyStopper:
    # From https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch.
    # Adapted for F1 score: should stop when is not increasing anymore.
    # Original one is designed for losses, stop when it does not decrease anymore.
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_f1 = 0

    def early_stop(self, f1):
        if f1 > self.max_validation_f1:
            self.max_validation_f1 = f1
            self.counter = 0
        elif f1 < (self.max_validation_f1 + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

@ex.config
def config():
    logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')
    device = 'cpu'  # DEFAULT_DEVICE
    iterations = 500000
    resume_iteration = None
    checkpoint_interval = 1000
    train_on = 'MAESTRO'
    model = "HPPNetDDD"
    preload_dataset = True

    batch_size = 4
    sequence_length = SAMPLE_RATE*5

    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = None

    convblock_length = 3
    add_dilated_convblock = True
    validation_length = sequence_length
    validation_interval = 500
    n_dilated_conv_layers = 3

    ex.observers.append(FileStorageObserver.create(logdir))


@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length, preload_dataset,
          model, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval, n_dilated_conv_layers, convblock_length, add_dilated_convblock):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    early_stopper = EarlyStopper(patience=15, min_delta=0.0005)
    stop_training = False
    train_groups, validation_groups = ['train'], ['validation']

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    if train_on == 'MAESTRO':
        dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length, device=device, preload=preload_dataset)
        validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length, device=device, preload=preload_dataset)
    else:
        dataset = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'], sequence_length=sequence_length, device=device, preload=preload_dataset)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length, device=device, preload=preload_dataset)

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, collate_fn=collating_function, num_workers=4)

    if resume_iteration is None:
        model = getattr(nnmodels, model)(n_dilated_conv_layers=n_dilated_conv_layers, convblock_length=convblock_length, add_dilated_convblock=add_dilated_convblock).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    progress_bar_metrics = {
        'loss': np.nan,
        "note_f1": np.nan,
    }

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    for i, batch in zip(loop, cycle(loader)):
        predictions, losses = model.run_on_batch(batch)

        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=i)

        progress_bar_metrics["loss"] = f"{loss.item():4.3f}"
        loop.set_postfix(progress_bar_metrics)

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                for key, value in evaluate(validation_dataset, model, pr_au_thresholds=None).items():
                    writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)

                    # Early stopping. Stop if f1 does not increase anymore
                    if key.replace(' ', '_') == "metric/note/f1":
                        if early_stopper.early_stop(np.mean(value)):
                            stop_training = True

                        progress_bar_metrics["note_f1"] = f"{np.mean(value):.4f}"
                        loop.set_postfix(progress_bar_metrics)
                    
            model.train()

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
        
        if stop_training:
            break
    
    torch.save(model, os.path.join(logdir, f"model-after_training.pt"))
