from .common import RandomChoices
from .vision import *


class AutoAugment(RandomChoice):
    """AutoAugment - Learning Augmentation Policies from Data
    * https://ai.googleblog.com/2018/06/improving-deep-learning-performance.html
    * https://github.com/DeepVoltaire/AutoAugment
    """

    @staticmethod
    def p_tran(p, t):
        if p <= 0:
            return []
        elif p >= 1:
            return [t]
        else:
            return [RandomApply(t, p=p)]

    @staticmethod
    def sub_trans(p0, p1):
        t = AutoAugment.p_tran(*p0) + AutoAugment.p_tran(*p1)
        if len(t) == 0:
            return PassThough()
        elif len(t) == 1:
            return t[0]
        else:
            return Compose(t)


class ImageNetAugment(AutoAugment):
    def __init__(self, fill=(128, 128, 128)):
        linspace_0_1 = np.linspace(0.0, 1.0, 11)

        posterize = lambda i: Posterize(np.round(np.linspace(8, 4, 10), 0).astype(np.int)[i])
        solarize = lambda i: Solarize(np.linspace(256, 0, 10)[i])
        rotate = lambda i: RandmonRotate(np.linspace(0, 30, 10)[i], fill=fill)
        color = lambda i: RandomAdjustColor(linspace_0_1[i], random='choice')
        sharpness = lambda i: RandomSharpness(linspace_0_1[i], random='choice')
        shearX = lambda i: RandomHorizontalShear(linspace_0_1[i] * 0.3, random='choice')
        contrast = lambda i: RandomContrast(linspace_0_1[i], random='choice')

        polices = (((0.4, posterize(8)), (0.6, rotate(9))),
                   ((0.6, solarize(5)),  (0.6, AutoContrast()),),
                   ((0.8, Equalize()),   (0.6, Equalize())),
                   ((0.6, posterize(7)), (0.6, posterize(6))),
                   ((0.4, Equalize()),   (0.2, solarize(4))),

                   ((0.4, Equalize()),   (0.8, rotate(8))),
                   ((0.6, solarize(3)),  (0.6, Equalize())),
                   ((0.8, posterize(5)), (1, Equalize())),
                   ((0.2, rotate(3)),    (0.6, solarize(8))),
                   ((0.6, Equalize()),   (0.4, posterize(6))),

                   ((0.8, rotate(8)),    (0,   color(0))),
                   ((0.4, rotate(9)),    (0.6, Equalize())),
                   ((0.0, Equalize()),   (0.8, Equalize())),
                   ((0.6, Invert()),     (1, Equalize())),
                   ((0.6, color(4)),     (1, contrast(8))),

                   ((0.8, rotate(8)),    (1,   color(2))),
                   ((0.8, color(8)),     (0.8, solarize(7))),
                   ((0.4, sharpness(7)), (0.6, Invert())),
                   ((0.6, shearX(5)),    (1, Equalize())),
                   ((0,   color(0)),     (0.6, Equalize())),

                   ((0.4, Equalize()),   (0.2, solarize(4))),
                   ((0.6, solarize(5)),  (0.6, AutoContrast())),
                   ((0.6, Invert()),     (1, Equalize())),
                   ((0.6, color(4)),     (1, contrast(8))),
                   ((0.8, Equalize()),   (0.6, Equalize()))
                   )

        trans = [AutoAugment.sub_trans(*p) for p in polices]
        super().__init__(trans)


class CIFAR10Augment(AutoAugment):
    def __init__(self, fill=(128, 128, 128)):
        linspace_0_1 = np.linspace(0.0, 1.0, 11)

        posterize = lambda i: Posterize(np.round(np.linspace(8, 4, 10), 0).astype(np.int)[i])
        solarize = lambda i: Solarize(np.linspace(256, 0, 10)[i])
        rotate = lambda i: RandmonRotate(np.linspace(0, 30, 10)[i], fill=fill)
        color = lambda i: RandomAdjustColor(linspace_0_1[i], random='choice')
        sharpness = lambda i: RandomSharpness(linspace_0_1[i], random='choice')
        shearY = lambda i: RandomVerticalShear(linspace_0_1[i] * 0.3, random='choice')
        contrast = lambda i: RandomContrast(linspace_0_1[i], random='choice')
        translateX = lambda i: RandomTranslate((linspace_0_1[i] * 0.5, 0), random='choice')
        translateY = lambda i: RandomTranslate((0, linspace_0_1[i] * 0.5), random='choice')
        brightness = lambda i: RandomBrightness(linspace_0_1[i], random='choice')

        polices = (((0.1, Invert()), (0.2, contrast(6))),
                   ((0.7, rotate(2)),  (0.3, translateX(9)),),
                   ((0.8, sharpness(1)),   (0.9, sharpness(3))),
                   ((0.5, shearY(8)), (0.7, translateY(9))),
                   ((0.5, AutoContrast()),   (0.9, Equalize())),

                   ((0.2, shearY(7)),   (0.3, posterize(7))),
                   ((0.4, color(3)),  (0.6, brightness(7))),
                   ((0.3, sharpness(9)), (0.7, brightness(9))),
                   ((0.6, Equalize()),    (0.5, Equalize())),
                   ((0.6, contrast(7)),   (0.6, sharpness(5))),

                   ((0.7, color(7)),    (0,   translateX(0))),
                   ((0.3, Equalize()),    (0.4, AutoContrast())),
                   ((0.4, translateY(3)),   (0.2, sharpness(6))),
                   ((0.9, brightness(6)),     (0.2, color(8))),
                   ((0.5, solarize(2)),     (0, Invert())),

                   ((0.2, Equalize()),    (0.6, AutoContrast())),
                   ((0.2, Equalize()),     (0.8, Equalize())),
                   ((0.9, color(9)),        (0.6, Equalize())),
                   ((0.8, AutoContrast()),    (0.2, solarize(8))),
                   ((0.1, brightness(3)),     (0,  color(0))),

                   ((0.4, solarize(5)),   (0.9, AutoContrast())),
                   ((0.9, translateY(9)),  (0.7, translateY(9))),
                   ((0.9, AutoContrast()),     (0.8, solarize(3))),
                   ((0.8, Equalize()),     (0.1, Invert())),
                   ((0.7, translateY(9)),     (0.9, AutoContrast()))
                   )

        trans = [AutoAugment.sub_trans(*p) for p in polices]
        super().__init__(trans)


class SVHNAugment(AutoAugment):
    def __init__(self, fill=(128, 128, 128)):
        linspace_0_1 = np.linspace(0.0, 1.0, 11)

        solarize = lambda i: Solarize(np.linspace(256, 0, 10)[i])
        rotate = lambda i: RandmonRotate(linspace_0_1[i] * 30, fill=fill)
        shearX = lambda i: RandomHorizontalShear(linspace_0_1[i] * 0.3, random='choice')
        shearY = lambda i: RandomVerticalShear(linspace_0_1[i] * 0.3, random='choice')
        contrast = lambda i: RandomContrast(linspace_0_1[i], random='choice')
        translateX = lambda i: RandomTranslate((linspace_0_1[i] * 0.5, 0), random='choice')
        translateY = lambda i: RandomTranslate((0, linspace_0_1[i] * 0.5), random='choice')

        polices = (((0.9, shearX(4)), (0.2, Invert())),
                   ((0.9, shearY(8)),  (0.7, Invert()),),
                   ((0.6, Equalize()),   (0.6, solarize(6))),
                   ((0.9, Invert()), (0.6, Equalize())),
                   ((0.6, Equalize()),   (0.9, rotate(3))),

                   ((0.9, shearX(4)),   (0.8, AutoContrast())),
                   ((0.9, shearY(8)),  (0.4, Invert())),
                   ((0.9, shearY(5)), (0.2, solarize(6))),
                   ((0.9, Invert()),    (0.8, AutoContrast())),
                   ((0.6, Equalize()),   (0.9, rotate(3))),

                   ((0.9, shearX(4)),    (0.3,   solarize(3))),
                   ((0.8, shearY(8)),    (0.7, Invert())),
                   ((0.9, Equalize()),   (0.6, translateY(6))),
                   ((0.9, Invert()),     (0.6, Equalize())),
                   ((0.3, contrast(3)),     (0.8, rotate(4))),

                   ((0.8, Invert()),    (0, translateY(2))),
                   ((0.7, shearY(6)),     (0.4, solarize(8))),
                   ((0.6, Invert()),        (0.8, rotate(4))),
                   ((0.3, shearY(7)),    (0.9, translateX(3))),
                   ((0.1, shearX(6)),     (0.6,  Invert())),

                   ((0.7, solarize(2)),   (0.6, translateY(7))),
                   ((0.8, shearY(4)),  (0.8, Invert())),
                   ((0.7, shearX(9)),     (0.8, translateY(3))),
                   ((0.8, shearY(5)),     (0.7, AutoContrast())),
                   ((0.7, shearX(2)),     (0.1, Invert()))
                   )

        trans = [AutoAugment.sub_trans(*p) for p in polices]
        super().__init__(trans)


class COCOAugment(AutoAugment):
    """
    Learning Data Augmentation Strategies for Object Detection
    https://arxiv.org/pdf/1906.11172.pdf
    """
    def __init__(self, fill=(128, 128, 128), version=0):
        linspace_0_1 = np.linspace(0.0, 1.0, 11)
        color = lambda i: RandomAdjustColor(linspace_0_1[i], random='choice')
        rotate = lambda i: RandmonRotate(linspace_0_1[i] * 30, fill=fill, random='choice')
        sharpness = lambda i: RandomSharpness(linspace_0_1[i], random='choice')
        translateX = lambda i: RandomTranslate((linspace_0_1[i] * 0.5, 0), border=fill, random='choice')
        translateY = lambda i: RandomTranslate((0, linspace_0_1[i] * 0.5), border=fill, random='choice')
        shearX = lambda i: RandomHorizontalShear(linspace_0_1[i] * 0.3, random='choice')
        shearY = lambda i: RandomVerticalShear(linspace_0_1[i] * 0.3, random='choice')
        cutout = lambda i: CutOut(max_size=int(200 * linspace_0_1[i]))
        translateYOnlyBBoxes = lambda i, p: ApplyMultiBBoxAugmentation(translateY(i), p)

        if version == 0:
            polices = (((0.6, translateX(4)), (0.8, Equalize())),
                       ((1, translateYOnlyBBoxes(2, 0.2)),  (0.8, cutout(8)),),
                       ((0, sharpness(8)), (0, shearX(1))),
                       ((1.0, shearY(2)), (1, translateYOnlyBBoxes(6, 0.6))),
                       ((0.6, rotate(10)),   (1.0, color(6))),
                       )
        else:
            raise NotImplementedError(f'version = {version}')

        trans = [AutoAugment.sub_trans(*p) for p in polices]
        super().__init__(trans)


class RandAug(RandomChoices):
    """RandAugment is from the paper https://arxiv.org/abs/1909.13719,"""

    def __init__(self, n=1, m=0.5):
        """
        Args:
            n: the number of augmentation transformations to apply sequentially
            m: shared magnitude across all augmentation operations
        """
        enhance_level = 0.9 * m + 0.1
        shear_level = 0.3 * m
        translate_level = 0.5 * m

        policies = [AutoContrast(),
                    Equalize(),
                    Invert(),
                    RandmonRotate(30 * m),
                    Posterize(int(4 * m)),
                    Solarize(int(255 * m)),
                    RandomAdjustColor(enhance_level),
                    RandomContrast(enhance_level),
                    RandomBrightness(enhance_level),
                    RandomBrightness(enhance_level),
                    RandomHorizontalShear(shear_level),
                    RandomVerticalShear(shear_level),
                    RandomTranslate((translate_level, 0), border=0),
                    RandomTranslate((0, translate_level), border=0),
                    CutOut(max_size=int(40 * m)),
                    SolarizeAdd(int(110 * m))]
        super().__init__(policies, k=n)
