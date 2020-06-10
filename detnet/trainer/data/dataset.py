
import collections
from random import randint
import torch
import torch.utils.data as tud


REPR_INDENT = ' ' * 2


class AttributeMissingMixin(object):
    """ A Mixin' to implement the 'method_missing' Ruby-like protocol. """
    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError as e:
            if attr.startswith('__'):
                raise e

            return self._attribute_missing(attr=attr)

    def _attribute_missing(self, attr):
        """ This method should be overridden in the derived class. """
        raise NotImplementedError(self.__class__.__name__ + " '_attribute_missing' method has not been implemented.")


class Dataset(tud.Dataset):
    repr_indent = REPR_INDENT

    def __add__(self, other):
        return ConcatDataset([self, other])

    def __radd__(self, other):
        if other == 0:
            return self  # for sum
        return self.__add__(other)

    def __rshift__(self, other):
        """transformed_dataset = dataset >> transform"""
        if not callable(other):
            raise RuntimeError('Dataset >> callable only!')
        return TransformedDataset(dataset=self, transform=other)

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += (REPR_INDENT + 'len: {}\n'.format(len(self)))
        return fmt_str

    def __getitem__(self, item):
        if isinstance(item, slice) or (not isinstance(item, str) and isinstance(item, collections.Iterable)):
            return Subset(self, item)

        return self.getitem(item)

    def getitem(self, idx):
        raise NotImplementedError(self.__class__.__name__ + " 'getitem' method has not been implemented.")


class ConcatDataset(Dataset, tud.ConcatDataset, AttributeMissingMixin):
    def __repr__(self):
        fmt_str = super().__repr__()
        for n, dataset in enumerate(self.datasets):
            if n >= 10:
                fmt_str += (REPR_INDENT + f'... {len(self.datasets) - n} more sub datasets ...\n')
                break

            dataset_str_lines = str(dataset).split('\n')
            dataset_str_lines = [s for s in dataset_str_lines if s]
            dataset_str_lines = ['-' * len(dataset_str_lines[0])] + dataset_str_lines
            dataset_str_lines = [REPR_INDENT + s for s in dataset_str_lines if s]
            fmt_str += '\n'.join(dataset_str_lines)
            if fmt_str[-1] != '\n':
                fmt_str += '\n'

        return fmt_str

    def getitem(self, idx):
        return tud.ConcatDataset.__getitem__(self, idx)

    def _attribute_missing(self, attr):
        """forward missing attr to dataset[0]"""
        return getattr(self.datasets[0], attr)

    def __iadd__(self, other):
        if isinstance(other, ConcatDataset):
            self.datasets.extend(other.datasets)
        else:
            self.datasets.append(other)
        return self

    def __add__(self, other):
        new_dataset = ConcatDataset(self.datasets.copy())
        new_dataset += other
        return new_dataset


class ExtendDataset(Dataset, AttributeMissingMixin):
    def __init__(self, dataset):
        self.dataset = dataset

    def _attribute_missing(self, attr):
        """forward missing attr to dataset"""
        return getattr(self.dataset, attr)

    def getitem(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        fmt_str = super().__repr__()
        dataset_str_lines = str(self.dataset).split('\n')
        dataset_str_lines = [s for s in dataset_str_lines if s]
        dataset_str_lines[0] = 'dataset: ' + dataset_str_lines[0]
        for i in range(1, len(dataset_str_lines)):
            dataset_str_lines[i] = ' ' * 9 + dataset_str_lines[i]
        dataset_str_lines = [REPR_INDENT + s for s in dataset_str_lines if s]
        fmt_str += '\n'.join(dataset_str_lines)
        if fmt_str[-1] != '\n':
            fmt_str += '\n'

        return fmt_str

    def evaluate(self, *args, **kwargs):
        dataset = self.dataset
        while isinstance(dataset, (ExtendDataset, ConcatDataset)):
            if dataset.__class__.evaluate != self.__class__.evaluate:
                # overloaded evaluate function
                break

            if isinstance(dataset, ConcatDataset):
                dataset = dataset.datasets[0]
            else:
                dataset = dataset.dataset

        evaluate_func = dataset.__class__.evaluate
        assert evaluate_func != self.__class__.evaluate
        return evaluate_func(self, *args, **kwargs)


class Subset(tud.Subset, ExtendDataset):
    """
    https://github.com/pytorch/vision/issues/369
    """
    def __init__(self, dataset, indices):
        if isinstance(indices, slice):
            indices = range(len(dataset))[indices]

        super().__init__(dataset, indices)


class RollSplitSet(Subset):
    def __init__(self, dataset, n_split):
        self.n_split = n_split
        self.i_split = 0
        self.end_split = len(dataset) - (len(dataset) % self.n_split)
        self.get_count = 0
        indices = slice(self.i_split, self.end_split, self.n_split)
        super().__init__(dataset, indices)

    def getitem(self, idx):
        item = super().getitem(idx)

        # use get count as indicator for rolling to next split
        self.get_count += 1
        if self.get_count == len(self):
            self.i_split += 1
            self.i_split %= self.n_split
            #print("RollSplitSet roll to next {}".format(self.i_split))
            indices = slice(self.i_split, self.end_split, self.n_split)
            self.indices = range(len(self.dataset))[indices]
            self.get_count = 0

        return item


class RandomDataset(ExtendDataset):
    def __init__(self, dataset, size):
        super().__init__(dataset)
        self.size = size

    def getitem(self, idx):
        idx = randint(0, len(self.dataset) - 1)
        return self.dataset[idx]

    def __len__(self):
        return self.size


class ShuffledDataset(Subset):
    def __init__(self, dataset):
        indices = torch.randperm(len(dataset))
        super().__init__(dataset, indices)


class WeightedRandomDataset(RandomDataset):
    def __init__(self, dataset, weights, num_samples, replacement=True):
        super().__init__(dataset, num_samples)
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement
        self.indices = []

    def getitem(self, idx):
        if not self.indices:
            self.indices = torch.multinomial(self.weights, self.size, self.replacement).tolist()
        idx = self.indices.pop(0)
        return self.dataset[idx]


class EchoingDataset(ExtendDataset):
    """cache the sample when reading and prepossessing data is too slow
    :see:
    Faster Neural Network Training with Data Echoing
    https://arxiv.org/pdf/1907.05550.pdf
    """
    def __init__(self, dataset, buffer_size, echo_ratio=2):
        super().__init__(dataset)
        self.buffer_size = buffer_size
        self.buffer = list()
        self.echo_ratio = echo_ratio

    def __len__(self):
        return len(self.dataset) * self.echo_ratio

    def getitem(self, idx):
        if len(self.buffer) >= self.buffer_size:
            idx = randint(0, len(self.buffer) - 1)
            return self.buffer.pop(idx)

        sample = super().getitem(idx % len(self.dataset))
        self.buffer.append(sample)
        return sample


class TransformedDataset(ExtendDataset):
    def __init__(self, dataset, transform):
        super().__init__(dataset)
        self.transform = transform

    def getitem(self, index):
        sample = self.dataset[index]
        return self.transform(sample)

    def __repr__(self):
        fmt_str = super().__repr__()
        tmp = '\n    Transforms: '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__str__().replace('\n', '\n' + ' ' * (len(tmp)-1)))
        return fmt_str
