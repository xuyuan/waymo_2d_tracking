import os
import sys
import shutil
import traceback
from pathlib import Path
import time
from datetime import datetime
import yaml
import argparse
import socket
import warnings
import random

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader, default_collate, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.cuda.amp import GradScaler
import torchvision.utils as vutils

from tqdm import tqdm, trange

from .optim import create_optimizer
from .test import inference
from .predictions import Predictions
from .mixup import mixup, cut_mix, f_mix
from .utils import choose_device, get_num_workers, arg2bool
from .data import split_dataset


def targets_to_cuda(targets, non_blocking=True):
    if isinstance(targets, torch.Tensor):
        targets = targets.cuda(non_blocking=non_blocking)
    elif isinstance(targets, list):
        targets = [targets_to_cuda(t, non_blocking=non_blocking) for t in targets]
    elif isinstance(targets, dict):
        targets = {k: targets_to_cuda(v, non_blocking=non_blocking) for k, v in targets.items()}
    elif isinstance(targets, (int, float, str)):
        return targets
    else:
        raise NotImplementedError(type(targets))
    return targets


def optimizer_cpu_state_dict(optimizer):
    # save cuda RAM
    optimizer_state_dict = optimizer.state_dict()

    dict_value_to_cpu = lambda d: {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                   for k, v in d.items()}

    if 'optimizer_state_dict' in optimizer_state_dict:
        #  FP16_Optimizer
        cuda_state_dict = optimizer_state_dict['optimizer_state_dict']
    else:
        cuda_state_dict = optimizer_state_dict

    if 'state' in cuda_state_dict:
        cuda_state_dict['state'] = {k: dict_value_to_cpu(v)
                                    for k, v in cuda_state_dict['state'].items()}

    return optimizer_state_dict


def load_weights(net, filename, strict=False):
    print('load weights {}'.format(filename))
    data = torch.load(filename, map_location='cpu')
    try:
        incompatible_keys = net.load_state_dict(data['state_dict'], strict=strict)
        print('loaded, incompatible_keys:', incompatible_keys)
    except RuntimeError as e:
        if strict:
            raise e
        else:
            warnings.warn(str(e))
    return net


def init_distributed(local_rank, device):
    if local_rank is not None:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        device = torch.device('cuda', local_rank)
        print('torch.distributed.init_process_group', device)
    else:
        device = choose_device(device)
    return device


def save_model(model, filename):
    if callable(getattr(model, 'save', None)):
        model.save(filename)
    else:
        torch.save(model, filename)


class Trainer(object):
    def __init__(self, args):
        """
        :param args: parsed results of `ArgumentParser`
        """
        try:
            self.args = args
            if args.local_rank is not None:
                assert args.device in ('auto', 'cuda')
                torch.cuda.set_device(args.local_rank)
                args.device = 'cuda'  # only support GPU

            self.device = init_distributed(args.local_rank, args.device)

            if args.local_rank is not None and args.local_rank > 0:
                # wait main_process finish initialization
                print(f'#{args.local_rank} wait main_process finish initialization', flush=True)
                torch.distributed.barrier()

        except Exception as e:
            if args.local_rank is not None:
                warnings.warn(f"Exception from process {args.local_rank}")
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
            raise e

    def _init(self, device, model, datasets, criterion):
        """
        :param device:
        :param model: `torch.nn.Module` to be trained
        :param datasets: dict of datasets including 'train', 'valid', and 'test'
        :param criterion: callable loss function, returns dict of losses
        :return:
        """
        args = self.args
        self.use_cuda = device.type == 'cuda'
        self.amp = args.amp
        self.grad_scaler = GradScaler(enabled=self.amp)
        assert self.amp is False  # pytorch 1.6 is required!

        self.is_main_process = args.local_rank is None or args.local_rank == 0
        self.datasets = datasets

        log_dir = Path(args.log_dir)
        self.swa_dir = Path("swa")

        # iteration counters
        self.iteration = 0
        self.start_epoch = 0
        self.min_epoch_loss = float('inf')
        self.max_metric_score = 0
        optimizer_state = None
        lr_scheduler_state = None
        amp_state = None

        if self.is_main_process:
            self.init_time = time.time()
            # print args
            args_yaml = yaml.dump((vars(args)))
            terminal_columns = shutil.get_terminal_size().columns
            self.println("=" * terminal_columns)
            self.println(args_yaml + ("=" * terminal_columns))
            self.swa_dir.mkdir(exist_ok=True)

        if args.weights:
            load_weights(model, args.weights)

        if args.resume:
            self.println('resume checkpoint ...')
            resume_checkpoint = torch.load(args.resume_checkpoint_file, map_location=lambda storage, loc: storage)
            if not args.weights:
                load_weights(model, resume_checkpoint['model_file'])
            self.start_epoch = resume_checkpoint['epoch']
            self.min_epoch_loss = resume_checkpoint.get('min_epoch_loss', self.min_epoch_loss)
            self.max_metric_score = resume_checkpoint.get('max_metric_score', self.max_metric_score)
            self.iteration = resume_checkpoint['iteration']
            optimizer_state = resume_checkpoint['optimizer']
            lr_scheduler_state = resume_checkpoint['lr_scheduler']
            amp_state = resume_checkpoint.get('amp', None)
            self.println('resume epoch {} iteration {}'.format(self.start_epoch, self.iteration))

        self.mix_epochs = args.no_mix_epochs if args.no_mix_epochs > 1.0 else (1 - args.no_mix_epochs) * args.max_epochs

        self.model = model
        lr_scheduler_args = vars(args)
        lr_scheduler_args['findlr_max_steps'] = len(datasets['train']) // args.batch_size
        self.net, self.optimizer, self.lr_scheduler, self.criterion = create_optimizer(device, model, criterion,
                                                                                       args.optim, args.learning_rate, args.weight_decay, args.momentum,
                                                                                       args.apex_opt_level,
                                                                                       optimizer_state=optimizer_state,
                                                                                       no_bn_wd=args.no_bn_wd,
                                                                                       local_rank=args.local_rank,
                                                                                       sync_bn=args.sync_bn,
                                                                                       lr_scheduler_args=lr_scheduler_args,
                                                                                       lr_scheduler_state=lr_scheduler_state,
                                                                                       lookahead=args.lookahead,
                                                                                       amp_state=amp_state,
                                                                                       cudnn_benchmark=args.cudnn_benchmark)

        if self.amp and isinstance(self.net, torch.nn.DataParallel) and torch.cuda.device_count() > 1:
            self.println('Attention! AMP + DataParallel requires autocast as part of model. DistributedDataParallel is recommended for mulit GPUs')

        self.checkpoints_folder = log_dir / 'checkpoints'
        self.checkpoints_folder.mkdir(parents=True, exist_ok=True)

        self.data_loaders = {k: self.create_dataset_loaders(d, k) for k, d in datasets.items() if k != 'test'}
        datasets_text = '\n '.join(['{} {}'.format(k, v) for k, v in datasets.items()])
        self.println("datasets:\n")
        self.println(datasets_text)

        self.tb_writer = None
        if self.is_main_process:
            try:
                from tensorboardX import SummaryWriter
            except:
                from torch.utils.tensorboard import SummaryWriter

            print("logging into {}".format(log_dir))
            self.tb_writer = SummaryWriter(str(log_dir))
            with (log_dir / 'args.yml').open('w') as f:
                f.write(args_yaml)

            # if not use_fp16:
            if args.write_graph:
                # write graph
                with torch.no_grad():
                    images = next(iter(self.data_loaders['train']))['input']
                    images = images.to(device)
                    model.trace_mode = True
                    self.tb_writer.add_graph(model, images)
                    model.trace_mode = False

    def println(self, *args, verbose=0, **kwargs):
        if self.is_main_process and self.args.verbose >= verbose:
            with tqdm.external_write_mode():
                print(*args, **kwargs)

    def get_num_workers(self):
        """return num of cpu workers per node"""
        num_workers = get_num_workers(self.args.jobs)
        if self.args.local_rank is not None:
            ngpus_per_node = torch.cuda.device_count()
            num_workers = num_workers // ngpus_per_node
        return num_workers

    def create_dataset_loaders(self, dataset, mode):
        args = self.args
        num_workers = self.get_num_workers()
        batch_size = args.batch_size
        if mode == 'valid' and args.validation_batch_size:
            batch_size = args.validation_batch_size

        sampler = None
        shuffle = mode == 'train'
        drop_last = mode == 'train'
        if not isinstance(dataset, IterableDataset):
            if args.local_rank is not None:
                sampler = DistributedSampler(dataset, shuffle=shuffle)
                shuffle = False
                if not args.batch_size_per_gpu:
                    ngpus_per_node = torch.cuda.device_count()
                    batch_size = batch_size // ngpus_per_node

        collate_fn = getattr(dataset, 'collate_fn', default_collate)
        data_loader = DataLoader(dataset, batch_size, shuffle=shuffle, sampler=sampler,
                                 num_workers=num_workers, drop_last=drop_last,
                                 collate_fn=collate_fn, pin_memory=self.use_cuda)

        return data_loader

    def preprocess_batch(self, batch):
        if isinstance(batch, dict):
            inputs = batch.pop('input', None)
            targets = batch
        elif isinstance(batch, (list, tuple)):
            inputs, targets = batch
        else:
            raise NotImplementedError(type(batch))
        if inputs is None:
            warnings.warn(f'no input, skip (data in batch are {batch.keys()})')
            assert False

        if self.use_cuda:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets_to_cuda(targets)
        return inputs, targets

    def run_epoch(self, epoch, phase):
        data_loader = self.data_loaders[phase]
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)
            # loss counters
        epoch_loss_dict = {}

        is_train = phase == 'train'
        if is_train:
            self.optimizer.zero_grad()
            if self.tb_writer:
                self.tb_writer.add_scalar('learning_rate', self.get_lr(), epoch)

        self.net.train(is_train)
        torch.set_grad_enabled(is_train)

        desc = f"Epoch {epoch} {phase}"
        if self.args.local_rank is not None:
            desc = f"[{self.args.local_rank}]" + desc

        _tqdm = lambda disable: tqdm(data_loader, desc=desc, unit="images", unit_scale=data_loader.batch_size,
                                     leave=False, disable=disable, mininterval=10, smoothing=1)
        pbar_print_froced = False
        pbar = _tqdm(None)

        if epoch == self.start_epoch + 1 and pbar.disable:
            pbar_print_froced = True
            pbar = _tqdm(False)

        it = 0
        # for logging images
        min_loss_in_epoch = float("inf")
        max_loss_in_epoch = 0
        batch_of_min_loss_in_epoch = None
        batch_of_max_loss_in_epoch = None

        for batch in pbar:
            inputs, targets = self.preprocess_batch(batch)

            criterion = self.criterion
            if phase == 'train' and epoch < self.mix_epochs:
                if self.args.mixup > 0 and random.random() < self.args.mix:
                    inputs, criterion = mixup(inputs, alpha=self.args.mixup, criterion=criterion)
                if self.args.cut_mix > 0 and random.random() < self.args.mix:
                    inputs, criterion = cut_mix(inputs, alpha=self.args.cut_mix, criterion=criterion)
                if self.args.f_mix > 0 and random.random() < self.args.mix:
                    inputs, criterion = f_mix(inputs, alpha=self.args.f_mix, criterion=criterion)

            # forward
            min_batch_size = len(inputs)

            #with autocast(self.amp):
            outputs = self.net(inputs)
            losses = criterion(outputs, targets)

            # compute overall loss if multi losses is returned
            if isinstance(losses, dict):
                if 'All' not in losses:
                    losses['All'] = sum(losses.values())
            elif isinstance(losses, torch.Tensor):
                losses = dict(All=losses)
            else:
                raise RuntimeError(type(losses))
            loss = losses['All']

            optimize_step = False
            if phase == 'train':
                if self.args.flooding > 0:
                    # "Do We Need Zero Training Loss After Achieving Zero Training Error?"
                    # https://github.com/takashiishida/flooding
                    loss = (loss - self.args.flooding).abs() + self.args.flooding

                loss = loss / self.args.gradient_accumulation
                loss = self.grad_scaler.scale(loss)
                self.optimizer.backward(loss)
                if self.iteration % self.args.gradient_accumulation == 0:
                    optimize_step = True
                    it += self.args.gradient_accumulation
                    if self.args.clip_grad_norm > 0 or self.args.clip_grad_value > 0:
                        # Unscales the gradients of optimizer's assigned params in-place
                        self.grad_scaler.unscale_(self.optimizer)
                        if self.args.clip_grad_norm > 0:
                            clip_grad_norm_(self.net.parameters(), self.args.clip_grad_norm)
                        if self.args.clip_grad_value > 0:
                            clip_grad_value_(self.net.parameters(), self.args.clip_grad_norm)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.optimizer.zero_grad()
                self.iteration += 1
                if self.lr_scheduler and self.lr_scheduler.name == 'findlr':
                    self.lr_scheduler.step(self.iteration)
                    if self.tb_writer:
                        self.tb_writer.add_scalar('learning_rate', self.get_lr(), self.iteration)
            elif phase == 'valid':
                it += 1

            if self.args.local_rank is not None:
                # sync loss between processes
                world_size = torch.distributed.get_world_size()
                for l in losses.values():
                    torch.distributed.all_reduce(l)
                    l /= world_size

            batch_loss_dict = {k: v.item() for k, v in losses.items()}

            min_batch_loss_scale = min_batch_size / data_loader.batch_size  # in case min_batch_size < batch_size (during validation)
            epoch_loss_dict = {k: epoch_loss_dict.get(k, 0) + v * min_batch_loss_scale for k, v in
                               batch_loss_dict.items()}

            if not self.is_main_process:
                continue
            # Below are logging in optimization step
            if self.tb_writer and optimize_step and self.args.log_loss_interval > 0 and self.iteration % self.args.log_loss_interval == 0:
                # tb_writer.add_scalars('Loss', batch_loss_dict, iteration)
                for k, v in batch_loss_dict.items():
                    self.tb_writer.add_scalar(phase + '/Loss/' + k, v, epoch)

            batch_loss = batch_loss_dict['All']
            if batch_loss < min_loss_in_epoch:
                min_loss_in_epoch = batch_loss
                batch_of_min_loss_in_epoch = (inputs, targets)
            if batch_loss > max_loss_in_epoch:
                max_loss_in_epoch = batch_loss
                batch_of_max_loss_in_epoch = (inputs, targets)

            if it > 0 and not pbar.disable:
                # update the progress bar
                scalars = {k: "%.03f" % (v / it) for k, v in epoch_loss_dict.items()}
                scalars['lr'] = self.get_lr()
                scalars['gRAM'] = f'{torch.cuda.memory_allocated() // 1000000}+{torch.cuda.memory_cached() // 1000000}'
                pbar.set_postfix(scalars, refresh=False)

                if pbar_print_froced:  # was force to print
                    if pbar.last_print_t - pbar.start_t > (self.args.verbose + 1) * 60:
                        pbar.disable = True

        epoch_loss_dict = {k: v / it for k, v in epoch_loss_dict.items()}
        if self.is_main_process and self.tb_writer:
            if self.args.log_images:
                name_batch = {"min_loss": batch_of_min_loss_in_epoch, "max_loss": batch_of_max_loss_in_epoch}
                for name, batch in name_batch.items():
                    if batch is not None:
                        images = self.visualize_batch(*batch)
                        images_grid = vutils.make_grid(images, normalize=False)
                        self.tb_writer.add_image('/'.join([phase, name]), images_grid, epoch)

            # scalars = {phase + k: v for k, v in epoch_loss_dict.items()}
            # tb_writer.add_scalars('EpochLoss', scalars, epoch)
            for k, v in epoch_loss_dict.items():
                self.tb_writer.add_scalar(phase + '/EpochLoss/' + k, v, epoch)

        return epoch_loss_dict['All']

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def save_swa_model(self):
        """save model for swa"""
        if self.args.swa_start == -1:
            return
        swa_start = self.args.swa_start if self.args.swa_start >= 0 else self.args.max_epochs + self.args.swa_start
        swa_start = self.current_epoch - swa_start
        if swa_start >= 0 and swa_start % self.args.swa_freq == 0:
            model_filename = self.swa_dir / (str(self.current_epoch) + '.model.pth')
            save_model(self.model, model_filename)

    def save_checkpoint(self, model_filename, checkpoint_filename=None):
        self.println(f'save checkpoint {model_filename} @ {self.current_epoch} epoch', file=sys.stderr)

        if not checkpoint_filename:
            checkpoint_filename = model_filename
        model_filename = str(self.checkpoints_folder / model_filename) + '.model.pth'
        checkpoint_filename = str(self.checkpoints_folder / checkpoint_filename) + '.checkpoint.pth'

        save_model(self.model, model_filename)

        #optimizer_state_dict = optimizer_cpu_state_dict(self.optimizer)

        data = {'epoch': self.current_epoch,
                'min_epoch_loss': self.min_epoch_loss,
                'max_metric_score': self.max_metric_score,
                'iteration': self.iteration,
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'model_file': model_filename,
                'args': self.args
                }
        if self.args.apex_opt_level:
            from apex import amp
            data['amp'] = amp.state_dict()
        torch.save(data,  checkpoint_filename)

        checkpoint_saved = Path(checkpoint_filename)
        last_checkpoint_file = self.checkpoints_folder / 'last.checkpoint'
        if last_checkpoint_file.exists():
            last_checkpoint_file.unlink()
        last_checkpoint_file.symlink_to(checkpoint_saved.relative_to(self.checkpoints_folder))
        self.last_saved_epoch = self.current_epoch

    def run(self, model, datasets, criterion, report_intermediate_result=None):
        try:
            if self.args.local_rank is not None and self.args.local_rank == 0:
                # notify other processes that main process initialized
                print(f'#{self.args.local_rank} notify other processes that main process initialized', flush=True)
                torch.distributed.barrier()

            self._init(self.device, model, datasets, criterion)
            print(f'#{self.args.local_rank} process initialized', flush=True)
            self._run(report_intermediate_result)
        finally:
            traceback.print_exc(file=sys.stdout)

            if self.is_main_process:
                if self.last_saved_epoch < self.current_epoch:
                    self.save_checkpoint("last")

                if self.args.swa_start != -1:
                    self.println('swa...')
                    from .swa import swa
                    swa(self.model.load, self.swa_dir, self.checkpoints_folder / "swa.model.pth", self.args.device, self.data_loaders['train'])

                if self.use_cuda:
                    self.println(f'Max GPU RAM usage: {torch.cuda.max_memory_allocated() // 1000000}+{torch.cuda.max_memory_cached() // 1000000}')
                self.println('Valid Loss = {}'.format(self.min_epoch_loss))
                self.println('Metric Score = {}'.format(self.max_metric_score))

                try:
                    hparam_dict = {k: v for k, v in vars(self.args).items() if k not in ('log_dir', 'local_rank') and v is not None}
                    metric_dict = {'metric_score': self.max_metric_score, 'valid_loss': self.min_epoch_loss}
                    self.tb_writer.add_hparams(hparam_dict, metric_dict)
                except Exception as e:
                    warnings.warn(f'failed to write tb hparams: {e}')
                self.tb_writer.close()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

    def _run(self, report_intermediate_result=None):
        self.println('Training', repr(self.model), 'Epochs:', self.start_epoch, '/', self.args.max_epochs, 'learning_rate:', self.get_lr(), verbose=3)
        if self.args.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        seed = self.args.random_seed
        if self.args.local_rank is not None:
            seed += self.args.local_rank
        torch.manual_seed(self.args.random_seed)
        np.random.seed(seed)
        random.seed(seed)

        pbar_epoch = trange(self.start_epoch + 1, self.args.max_epochs + 1, initial=self.start_epoch,
                            unit="epoch", disable=not self.is_main_process)
        self.last_saved_epoch = 0
        early_stopping = False
        for epoch in pbar_epoch:
            self.current_epoch = epoch
            epoch_state = {'lr': self.get_lr()}
            for phase in self.data_loaders:
                if phase == 'valid' and epoch % self.args.validation_interval != 0:
                    continue

                epoch_loss = self.run_epoch(epoch, phase)

                self.lr_scheduler.step()
                evaluation = None
                if phase == 'valid' or 'valid' not in self.data_loaders:
                    if self.lr_scheduler.name == 'plateau':
                        self.lr_scheduler.step(metrics=epoch_loss)

                    if epoch % self.args.validation_interval == 0:
                        if 'test' in self.datasets:
                            evaluation = self.test()

                if self.is_main_process:
                    # Below are processing between epoch, e.g. save checkpoints, logging, etc.

                    if evaluation is not None:
                        epoch_state['metric'] = metric_score = float(evaluation['score'])
                        if report_intermediate_result:
                            report_intermediate_result(metric_score)

                        self.tb_writer.add_scalar('test/metric', metric_score, epoch)
                        for k, v in evaluation.items():
                            if isinstance(v, dict) and 'score' in v:
                                self.tb_writer.add_scalar('test/' + k.replace(' ', '_'), v['score'], epoch)

                        if metric_score > self.max_metric_score:
                            self.max_metric_score = metric_score
                            self.save_checkpoint("best_metric")

                    if phase == 'valid' or 'valid' not in self.data_loaders:
                        if self.min_epoch_loss > epoch_loss:
                            self.min_epoch_loss = epoch_loss
                            self.save_checkpoint('best_loss')

                    if self.args.checkpoints_interval > 0 and epoch % self.args.checkpoints_interval == 0 and self.last_saved_epoch < self.current_epoch:
                        self.save_checkpoint("last")

                    epoch_state[phase + '_loss'] = epoch_loss

                early_stopping = (self.get_lr() < self.args.min_learning_rate)

            if self.is_main_process:
                self.save_swa_model()
                epoch_state['time'] = datetime.now().strftime('%d%b%H:%M')
                epoch_state['min_loss'] = self.min_epoch_loss
                epoch_state['max_metric_score'] = self.max_metric_score
                if self.use_cuda:
                    epoch_state['gRAM'] = f'{torch.cuda.memory_allocated() // 1000000}+{torch.cuda.memory_cached() // 1000000}'
                pbar_epoch.set_postfix(epoch_state, refresh=False)

                # save results so far
                epoch_state['epoch'] = epoch
                epoch_state['init_time'] = self.init_time
                epoch_state['duration'] = time.time() - self.init_time
                if torch.distributed.is_initialized():
                    epoch_state['distributed_world_size'] = torch.distributed.get_world_size()
                else:
                    epoch_state['distributed_world_size'] = 1
                with open(self.checkpoints_folder.parent / "results.yml", 'w') as f:
                    yaml.dump(epoch_state, f)

            if early_stopping:
                print('early stopping')
                break

            if self.args.lr_scheduler == 'findlr':
                break

    def predict(self, dataset):
        """
        make prediction for all the samples in the dataset
        :param dataset:
        :return:
        """
        self.model.eval()
        if self.args.test_processes > 1:
            self.model.share_memory()

        num_workers = self.get_num_workers()
        classnames = getattr(self.model, 'classnames', None)
        if self.args.local_rank is None:
            detections, inference_time = inference(self.model.predict, dataset, sub_processes=self.args.test_processes, batch_size=self.args.test_batch_size, num_dataloader_workers=num_workers)
            return Predictions([detections], classnames)
        else:
            # distributed test
            world_size = torch.distributed.get_world_size()
            batch_size = self.args.test_batch_size // world_size
            if self.args.test_batch_size > 0:
                batch_size = max(1, batch_size)
            sub_test_dataset = split_dataset(dataset, world_size)[self.args.local_rank]
            output_root = self._test_temp_folder()
            if self.args.local_rank == 0:
                shutil.rmtree(output_root, ignore_errors=True)
                output_root.mkdir(parents=True)
            torch.distributed.barrier()  # synchronizes all processes.
            detections, inference_time = inference(self.model.predict, sub_test_dataset, output_root=output_root, process_id=self.args.local_rank, sub_processes=self.args.test_processes,
                                                   batch_size=batch_size, num_dataloader_workers=num_workers)
            torch.distributed.barrier()  # synchronizes all processes.
            if self.args.local_rank == 0:
                # evaluation in main process only
                detection_files = [str(f)[:-4] for f in output_root.glob('*.dir')]
                return Predictions(detection_files, classnames)
            else:
                return None

    def _test_temp_folder(self):
        return Path('/dev/shm') / Path(self.args.log_dir).name / "test"

    def test(self):
        """
        * model `net` has `predict` function to be implemented
        * test dataset has `evaluate` function
        """
        test_dataset = self.datasets['test']
        predictions = self.predict(test_dataset)

        evaluation = None
        if predictions is not None:  # main process
            evaluation = self.eval(test_dataset, predictions)

        if self.args.local_rank is not None:
            torch.distributed.barrier()  # synchronizes all processes.

        return evaluation

    def eval(self, test_dataset, predictions):
        num_workers = get_num_workers(self.args.jobs)
        evaluation = test_dataset.evaluate(predictions, num_processes=num_workers)
        predictions.remove()  # TODO: it doesn't remove tmp root directory
        shutil.rmtree(self._test_temp_folder(), ignore_errors=True)
        return evaluation

    def visualize_batch(self, inputs, targets):
        raise NotImplementedError()


class ArgumentParser(object):
    def __init__(self, description=__doc__):
        self.parser = argparse.ArgumentParser(description=description,
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                              fromfile_prefix_chars='@')
        group = self.parser.add_argument_group('general options')
        group.add_argument("comment", nargs='*')
        group.add_argument('--validation-interval', type=int, default=5, help='interval of epochs for validation')
        group.add_argument("--mixup", type=float, default=-1, help='MixUp alpha hyperparameter, negative for disabling')
        group.add_argument("--cut-mix", type=float, default=-1, help='CutMix alpha hyperparameter, negative for disabling')
        group.add_argument("--f-mix", type=float, default=-1, help='FMix alpha hyperparameter, negative for disabling')
        group.add_argument("--mix", type=float, default=0.5, help='probabilities to apply MixUp/CutMix/FMix')
        group.add_argument('--no-mix-epochs', type=float, default=0.1,
                           help='Disable MixUp/CutMix/FMix training if enabled in the last N (or %%) epochs.')
        group.add_argument('--apex-opt-level', default=None, choices=('O0', 'O1', 'O2', 'O3'),
                           help='different pure and mixed precision modes from apex')
        group.add_argument('--amp', action='store_true', help='use Automatic Mixed Precision')
        group.add_argument('--sync-bn', action='store_true', help='enabling sync BatchNorm')
        group.add_argument('--detect-anomaly', action='store_true', help='enable anomaly detection to find the operation that failed to compute its gradient')
        group.add_argument('--random-seed', type=int, default=0, help='seed the RNG')
        group.add_argument('--test-processes', type=int, default=1, help='number of test processes each node')

        group = self.parser.add_argument_group('options of devices')
        group.add_argument('--device', default='auto', choices=['cuda', 'cpu'], help='running with cpu or cuda')
        group.add_argument('--cudnn-benchmark', default=True, type=arg2bool, help='enable cudnn benchmark')
        group.add_argument("--local_rank", type=int, default=None, help='args for torch.distributed.launch module')

        group = self.parser.add_argument_group('options of dataloader')
        group.add_argument('--batch-size-per-gpu', action='store_true', help='set batch_size value per gpu instead of total')
        group.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
        group.add_argument('--validation-batch-size', type=int, help='Batch size for validation, default to same as training batch size')
        group.add_argument('--test-batch-size', type=int, default=0, help='Batch size for test, 0 means no batch')
        group.add_argument('--jobs', default=-2, type=int,
                           help='How many subprocesses to use for data loading. ' +
                                'Negative or 0 means number of cpu cores left')

        group = self.parser.add_argument_group('options of optimizer')
        group.add_argument("--optim", default='sgd',
                           choices=['sgd', 'adam', 'adamw', 'RMSprop', 'radam', 'fused_adam', 'adabound', 'adaboundw', 'novograd'],
                           help='choices of optimization algorithms')
        group.add_argument('--max-epochs', default=100, type=int, help='Number of training epochs')
        group.add_argument('--learning-rate', default=1e-3, type=float, help='initial learning rate')
        group.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD optimizer')
        group.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay')
        group.add_argument('--no-bn-wd', action='store_true', help='Remove batch norm from weight decay')
        group.add_argument('--gradient-accumulation', type=int, default=1,
                           help='accumulate gradients over number of batches')
        group.add_argument('--clip-grad-norm', type=float, default=0,
                           help='clips gradient norm of model parameters.')
        group.add_argument('--clip-grad-value', type=float, default=0,
                           help='clips gradient value of model parameters.')
        group.add_argument('--lookahead', action='store_true', help='use Lookahead Optimizer: k steps forward, 1 step back')
        group.add_argument('--flooding', type=float, default=0,
                           help='intentionally prevents further reduction of the training loss when it reaches flooding level')

        group = self.parser.add_argument_group('options of learning rate scheduler')
        group.add_argument("--lr-scheduler", default='step', help='method to adjust learning rate',
                           choices=['plateau', 'step', 'cos', 'cosw', 'findlr', 'noam'])
        group.add_argument("--lr-scheduler-patience", type=int, default=20,
                           help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
        group.add_argument("--lr-scheduler-step-size", type=lambda s: [float(item) for item in s.split(',')], default=[0.4],
                           help='number of epochs of learning rate decay, or cycle for cos/cosw scheduler')
        group.add_argument("--lr-scheduler-gamma", type=float, default=0.1,
                           help='learning rate is multiplied by the gamma to decrease it')
        group.add_argument("--lr-scheduler-warmup", type=float, default=0.1,
                           help='The number/percentage of epochs to linearly increase the learning rate.')
        group.add_argument("--min-learning-rate", type=float, default=1e-9,
                           help='stop when learning rate is smaller than this value')

        group = self.parser.add_argument_group('options of logging and saving')
        group.add_argument('--resume', default=None, type=str, help='path of args.yml for resuming')
        group.add_argument('--resume-checkpoint-file', default=None, type=str,
                           help='path of checkpoint file for resuming, the "checkpoints/last.checkpoint.pth" is default')
        group.add_argument('--weights', type=str, help='pretrained weights')
        group.add_argument("--write-graph", action='store_true', help='visualize graph in tensorboard')
        group.add_argument('--log-images', action='store_true', help='save image samples each batch')
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        default_log_dir = os.path.join('runs', current_time + '_' + socket.gethostname())
        default_log_dir = os.getenv("TRAIN_LOG_DIR", default_log_dir)
        group.add_argument('--log-dir', type=str, default=default_log_dir, help='Location to save logs and checkpoints')
        group.add_argument('--log-loss-interval', type=int, default=0,
                           help='interval of iterations for loss logging, nonpositive for disabling')
        group.add_argument('--verbose', '-v', action='count', default=0, help='verbosity levels')
        group.add_argument('--checkpoints-interval', type=int, default=20,
                           help='interval of epochs for saving checkpoints')
        group.add_argument('--swa-start', type=int, default=-1, help='number of epochs before starting to apply SWA')
        group.add_argument('--swa-freq', type=int, default=1, help='number of epochs between subsequent updates of SWA running averages')

    def add_argument(self, *kargs, **kwargs):
        self.parser.add_argument(*kargs, **kwargs)

    def add_argument_group(self, *kargs, **kwargs):
        return self.parser.add_argument_group(*kargs, **kwargs)

    def set_defaults(self, **kwargs):
        self.parser.set_defaults(**kwargs)

    def parse_args(self, args=None, namespace=None):
        parsed_args = self.parser.parse_args(args=args, namespace=namespace)
        if parsed_args.resume:
            parsed_args = self.update_args_from_file(parsed_args, parsed_args.resume)
            # overwrite with command line again
            parsed_args = self.parser.parse_args(args=args, namespace=parsed_args)
            if getattr(parsed_args, 'resume_checkpoint_file', None) is None:
                # set default resume checkpoint file
                resume_dir = Path(parsed_args.resume).parent
                checkpoint_file = resume_dir / 'checkpoints' / 'last.checkpoint'
                setattr(parsed_args, 'resume_checkpoint_file', str(checkpoint_file))

        return parsed_args

    def update_args_from_file(self, args, filename):
        # for different version of PyYAML
        if hasattr(yaml, 'full_load'):
            args_var = yaml.full_load(Path(filename).open())
        else:
            args_var = yaml.load(Path(filename).open())

        for name in args_var:
            if name not in ('resume_checkpoint_file', 'weights', 'local_rank'):
                setattr(args, name, args_var[name])
        return args



