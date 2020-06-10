import random
import torchvision.transforms as tvt


def _compose_repr(self, args_string=''):
    format_string = self.__class__.__name__ + '('
    indent = ' ' * len(format_string)
    trans_strings = [repr(t).replace('\n', '\n' + indent) for t in self.transforms]
    if args_string:
        trans_strings.insert(0, args_string)

    format_string += (',\n'+indent).join(trans_strings)
    format_string += ')'
    return format_string


class Compose(tvt.Compose):
    def __repr__(self): return _compose_repr(self)

    def redo(self, sample):
        for t in self.transforms:
            sample = t.redo(sample)
        return sample


class RandomApply(Compose):
    def __init__(self, transforms, p=0.5):
        if not isinstance(transforms, (list, tuple)):
            transforms = [transforms]
        super().__init__(transforms)
        self.p = p
        self._applied = False

    def __repr__(self):
        format_string = 'p={}'.format(self.p)
        return _compose_repr(self, format_string)

    def __call__(self, sample):
        if self.p < random.random():
            return sample
        self._applied = True
        return super().__call__(sample)

    def redo(self, sample):
        if self._applied:
            return super().redo(sample)
        return sample


class ScheduledRandomApply(RandomApply):
    """RandomApply with changing probability"""
    def __init__(self, transforms, start_p=0, stop_p=1, step_p=0.01):
        super().__init__(transforms, p=start_p)
        if step_p > 0:
            assert stop_p > start_p

        self.min_p = min(start_p, stop_p)
        self.max_p = max(start_p, stop_p)
        self.step_p = step_p

    def __call__(self, sample):
        sample = super().__call__(sample)
        self.p = max(min(self.p + self.step_p, self.max_p), self.min_p)
        return sample


class RandomOrder(tvt.RandomOrder):
    def __repr__(self): return _compose_repr(self)


class RandomChoice(tvt.RandomChoice):
    def __repr__(self): return _compose_repr(self)

    def __call__(self, sample):
        self._trans = random.choice(self.transforms)
        return self._trans(sample)

    def redo(self, sample):
        return self._trans.redo(sample)


class RandomChoices(RandomChoice):
    """Apply k transformations randomly picked from a list
    """
    def __init__(self, transforms, k=1):
        super(RandomChoices, self).__init__(transforms)
        self.k = k

    def __repr__(self):
        format_string = 'k={}'.format(self.k)
        return _compose_repr(self, format_string)

    def __call__(self, sample):
        self._trans = Compose(random.sample(self.transforms, k=self.k))
        return self._trans(sample)

    def redo(self, sample):
        return self._trans.redo(sample)


class PassThough(object):
    def __call__(self, sample): return sample

    def __repr__(self): return self.__class__.__name__ + '()'


class FilterSample(object):
    def __init__(self, keep_keys):
        self.keep_keys = keep_keys

    def __call__(self, sample):
        return {k: v for k, v in sample.items() if k in self.keep_keys}

    def redo(self, sample):
        return self(sample)

    def __repr__(self):
        return self.__class__.__name__ + '(' + repr(self.keep_keys) + ')'


class ApplyOnly(object):
    def __init__(self, keys, transform):
        self.keys = keys
        self.transform = transform

    def __call__(self, sample):
        filtered_sample = {k: v for k, v in sample.items() if k in self.keys}
        excluded_sample = {k: v for k, v in sample.items() if k not in self.keys}
        sample = self.transform(filtered_sample)
        sample.update(excluded_sample)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(keys=' + repr(self.keys) + ', transform=' + repr(self.transform) + ')'
