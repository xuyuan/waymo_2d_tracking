
from functools import lru_cache
import pickle
import shelve
import time
from pathlib import Path
import multiprocessing as MP
import torch
from torch import multiprocessing
from torch.utils.data.dataloader import DataLoader, default_collate
from tqdm import tqdm
from .utils import choose_device, get_num_workers
from .predictions import Predictions
from .data import split_dataset


def inference_worker(kwargs):
    return _inference(**kwargs)


def inference(predict, samples, output_root=None, process_id=None, disable_tqdm=None, sub_processes=1, batch_size=0, num_dataloader_workers=0):
    """
    :param predict: callable(input) --> prediction
    :param samples: dataset or collection of samples
    :param output_root: path to save predictions
    :param process_id: id of process (used for print and saving predictions)
    :param disable_tqdm: disable progress bar
    :param sub_processes: spawn multi-processes within same GPU (via shared model)
    :param batch_size: process in batch with dataloader, 0 means no dataloader, otherwise samples has to be collectible
    :param num_dataloader_workers: num_workers of dataloader if batch_size > 0
    :return: predictions, inference_time
    """
    if sub_processes == 1:
        return _inference(predict=predict, samples=samples, output_root=output_root, process_id=process_id, disable_tqdm=disable_tqdm,
                          batch_size=batch_size, num_dataloader_workers=num_dataloader_workers)

    start_time = time.time()
    dataset_splits = split_dataset(samples, sub_processes)
    worker_args = [dict(predict=predict, samples=subset, output_root=output_root, process_id=process_id, disable_tqdm=(True if i != 0 else disable_tqdm), sub_processes_id=i,
                        batch_size=batch_size, num_dataloader_workers=num_dataloader_workers // sub_processes)
                   for i, subset in enumerate(dataset_splits)]

    multiprocessing_spawn = multiprocessing.get_context('spawn')  # necessary for sharing cuda data
    with multiprocessing_spawn.Pool(processes=sub_processes - 1) as pool:
        sub_ret = pool.map_async(inference_worker, worker_args[1:])
        predictions, t = inference_worker(worker_args[0])
        if output_root is not None:
            assert isinstance(predictions, str)
            predictions = [predictions]

        sub_ret.wait()
        for pred, t in sub_ret.get():
            if output_root is not None:
                predictions.append(pred)
            else:
                predictions.update(pred)

        finish_time = time.time()
        inference_time = finish_time - start_time
        return predictions, inference_time


def _inference(predict, samples, output_root=None, process_id=None, disable_tqdm=None, sub_processes_id=None, batch_size=0, num_dataloader_workers=0):
    assert callable(predict)
    desc = "inference"
    if process_id is not None:
        desc += " #{}".format(process_id)

    unit = "image"
    if batch_size > 0:
        collate_fn = getattr(samples, 'collate_fn', default_collate)
        samples = DataLoader(samples, batch_size, num_workers=num_dataloader_workers, collate_fn=collate_fn, pin_memory=True)
        if process_id is None or process_id == 0:
            print(f'DataLoader: num_workers={num_dataloader_workers}, batch_size={batch_size}')
        unit = f"x{batch_size}" + unit

    pbar = tqdm(samples, unit=unit, desc=desc, position=process_id, disable=disable_tqdm,
                mininterval=60, miniters=len(samples)//100, smoothing=1)

    detections = {}
    output_detection = None

    if output_root is not None:
        output_detection = str(output_root / "detections")
        if process_id is not None:
            output_detection += str(process_id)
        if sub_processes_id is not None:
            output_detection += ('.' + str(sub_processes_id))
        detections = shelve.open(output_detection, flag='n', protocol=pickle.HIGHEST_PROTOCOL)

    with torch.no_grad():
        start_time = time.time()
        for i, sample in enumerate(pbar):
            image_id = str(i)
            if isinstance(sample, dict):
                image_id = sample.get('image_id', image_id)
                inputs = sample.get('input', None)
            elif isinstance(sample, (list, tuple)):
                inputs, targets = sample
            elif torch.is_tensor(sample):
                inputs = sample

            d = predict(inputs)
            if torch.is_tensor(d):
                d = d.cpu()
            if image_id:
                if isinstance(image_id, list):
                    for j in range(len(image_id)):
                        detections[image_id[j]] = d[j]
                else:
                    detections[image_id] = d

        finish_time = time.time()
    inference_time = finish_time - start_time

    if output_detection is not None:
        detections.close()  # close shelve
        # return filename
        return output_detection, inference_time
    else:
        return detections, inference_time


class Tester(object):
    def __init__(self, create_model=None, device='auto', jobs=1, disable_tqdm=None, cudnn_benchmark=False):
        self.create_model = create_model if create_model else Tester.default_create_model
        self.device = device
        self.jobs = get_num_workers(jobs, choose_device(device))
        self.disable_tqdm = disable_tqdm
        self.cudnn_benchmark = cudnn_benchmark
        self.num_dataloader_workers = get_num_workers(-self.jobs) // self.jobs  # (#cpus - #jobs) / (#jobs)

    @staticmethod
    def default_create_model(filename):
        model = torch.load(filename)
        classes = None
        return model, classes

    @lru_cache(maxsize=1)
    def _create_or_load_model(self, model_file, cuda_device_id, use_half, sub_processes):
        if cuda_device_id is not None:
            torch.backends.cudnn.benchmark = self.cudnn_benchmark
            #torch.backends.cudnn.deterministic = True
            torch.cuda.set_device(cuda_device_id)
            use_cuda = True
        else:
            use_cuda = False

        model, classes = self.create_model(model_file)

        if use_cuda:
            # if self.jobs == 1 and sub_processes == 1 and batch_size > 1 and torch.cuda.device_count() > 1:
            #     print(f"using DataParallel to utilize {torch.cuda.device_count()} GPUs")
            #     model = torch.nn.DataParallel(model)

            model.cuda()
            if use_half:
                model.half()

        model.eval()
        if sub_processes > 1:
            model.share_memory()

        return model, classes

    def inference(self, model_file, dataset, output_root=None, process_id=None, cuda_device_id=None, use_half=False, sub_processes=1, batch_size=0):
        model, classes = self._create_or_load_model(model_file, cuda_device_id, use_half, sub_processes)
        det, inference_time = inference(model, dataset, output_root, process_id, self.disable_tqdm, sub_processes=sub_processes,
                                        batch_size=batch_size, num_dataloader_workers=self.num_dataloader_workers)
        return classes, det

    def inference_(self, kwargs, ret_q=None):
        process_id = kwargs.get('process_id', 0)
        if process_id >= torch.cuda.device_count():
            # wait to avoid GPU OOM due to cudnn.benchmark
            time.sleep((process_id - torch.cuda.device_count() + 1) * 10)

        ret = None
        try:
            ret = self.inference(**kwargs)
        except Exception as e:
            print(f'inference exception with args {kwargs}\n{e}')
        finally:
            if ret_q and ret is not None:
                ret_q.put(ret)

            return ret

    def test(self, model_file, dataset, output=None, data_balanced=False, sub_processes=1, batch_size=0, resume=None):
        if resume:
            resume_predictions = Predictions.open(resume)
            tested_samples = list(resume_predictions.keys())
            print(f"resuming {len(tested_samples)} tested samples")
            dataset.exclude(tested_samples)  # exclude must be in dataset implemented for resume
            print('new dataset:', dataset)

        device = choose_device(self.device)
        use_cuda = device.type == 'cuda'
        use_half = self.device == 'half'

        output_root = None
        if output:
            output_root = Path(output)
            output_root.mkdir(parents=True, exist_ok=True)

        detections = {}
        detection_files = []
        classnames = None

        num_processes = self.jobs
        if num_processes == 1:
            cuda_device_id = None
            if use_cuda:
                cuda_device_id = 0 if device.index is None else device.index
            classnames, r = self.inference(model_file, dataset, output_root, cuda_device_id=cuda_device_id, use_half=use_half, sub_processes=sub_processes, batch_size=batch_size)
            if isinstance(r, str):
                detection_files.append(r)
            elif isinstance(r, list):
                detection_files += r
            else:
                detections = r
        else:
            assert (num_processes > 1)
            if not output_root:
                raise RuntimeError("multi-processes test only possible with output as cache")
            print('start', num_processes, 'processes')
            dataset_splits = split_dataset(dataset, num_processes, balanced=data_balanced)
            worker_args = [dict(model_file=model_file, dataset=subset, process_id=i, output_root=output_root, use_half=use_half,  sub_processes=sub_processes, batch_size=batch_size)
                           for i, subset in enumerate(dataset_splits)]

            if use_cuda:
                n_gpu = torch.cuda.device_count()
                # distribute to GPUs
                if n_gpu > 0:
                    print('with {} GPUs'.format(n_gpu))
                    for i, arg in enumerate(worker_args):
                        arg['cuda_device_id'] = i % n_gpu

            sub_ret = []
            multiprocessing_spawn = multiprocessing.get_context('spawn')  # necessary for sharing cuda data
            # with multiprocessing_spawn.Pool(processes=num_processes) as pool:
            #     q = pool.map_async(self.inference_, worker_args)
            #     q.wait()
            #     for r in q.get():
            #         sub_ret.append(r)

            # start process with daemon == False
            q = multiprocessing_spawn.Queue()
            processes = [multiprocessing_spawn.Process(target=self.inference_, args=(kwargs, q)) for kwargs in worker_args]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            for p in processes:
                sub_ret.append(q.get(timeout=10))

            for cls, r in sub_ret:
                if classnames is None:
                    classnames = cls
                else:
                    assert classnames == cls

                if isinstance(r, str):
                    detection_files.append(r)
                elif isinstance(r, list):
                    detection_files += r
                else:
                    detections.update(r)

        predictions = Predictions([detections] + detection_files, classnames, mode='w')
        if resume:
            predictions.update(resume_predictions)
        if output_root:
            output_detection = output_root / "detections.pkl"
            print('saving', output_detection)
            predictions.save(output_detection)
            print('saved', output_detection)

        return predictions
