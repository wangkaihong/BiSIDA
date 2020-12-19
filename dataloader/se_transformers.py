import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import numbers
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
try:
    import accimage
except ImportError:
    accimage = None
import math
from utils import _get_inverse_affine_matrix

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def affine(img, angle, translate, scale, shear, resample=0, fillcolor=None):
    """Apply affine transformation on the image keeping image center invariant

    Args:
        img (PIL Image): PIL Image to be rotated.
        angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float or tuple or list): shear angle value in degrees between -180 to 180, clockwise direction.
        If a tuple of list is specified, the first value corresponds to a shear parallel to the x axis, while
        the second value corresponds to a shear parallel to the y axis.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter.
            See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    output_size = img.size
    center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] >= '5' else {}
    return img.transform(output_size, Image.AFFINE, matrix, resample, **kwargs)


class Tgt_Horizontal_Flip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2, lbl1, lbl2, transform_param1, transform_param2):
        if np.random.random() < self.p:
            img1 = F.hflip(img1)
            transform_param1[0] = True
            img2 = F.hflip(img2)
            transform_param2[0] = True
            lbl1 = F.hflip(lbl1)
            lbl2 = F.hflip(lbl2)
        else:
            transform_param1[0] = False
            transform_param2[0] = False
        return img1, img2, lbl1, lbl2, transform_param1, transform_param2

# class Tgt_RandomAffine():

#     def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor_rgb=(0,0,0), fillcolor_gray=255):
#         if isinstance(degrees, numbers.Number):
#             if degrees < 0:
#                 raise ValueError("If degrees is a single number, it must be positive.")
#             self.degrees = (-degrees, degrees)
#         else:
#             assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
#                 "degrees should be a list or tuple and it must be of length 2."
#             self.degrees = degrees

#         if translate is not None:
#             assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
#                 "translate should be a list or tuple and it must be of length 2."
#             for t in translate:
#                 if not (0.0 <= t <= 1.0):
#                     raise ValueError("translation values should be between 0 and 1")
#         self.translate = translate

#         if scale is not None:
#             assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
#                 "scale should be a list or tuple and it must be of length 2."
#             for s in scale:
#                 if s <= 0:
#                     raise ValueError("scale values should be positive")
#         self.scale = scale

#         if shear is not None:
#             if isinstance(shear, numbers.Number):
#                 if shear < 0:
#                     raise ValueError("If shear is a single number, it must be positive.")
#                 self.shear = (-shear, shear)
#             else:
#                 assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
#                     "shear should be a list or tuple and it must be of length 2."
#                 self.shear = shear
#         else:
#             self.shear = shear

#         self.resample = resample
#         self.fillcolor_rgb = fillcolor_rgb
#         self.fillcolor_gray = fillcolor_gray

#     @staticmethod
#     def get_params(degrees, translate, scale_ranges, shears, img_size):
#         """Get parameters for affine transformation

#         Returns:
#             sequence: params to be passed to the affine transformation
#         """
#         angle = np.random.uniform(degrees[0], degrees[1])
#         if translate is not None:
#             max_dx = translate[0] * img_size[0]
#             max_dy = translate[1] * img_size[1]
#             translations = (np.round(np.random.uniform(-max_dx, max_dx)),
#                             np.round(np.random.uniform(-max_dy, max_dy)))
#         else:
#             translations = (0, 0)

#         if scale_ranges is not None:
#             scale = np.random.uniform(scale_ranges[0], scale_ranges[1])
#         else:
#             scale = 1.0

#         if shears is not None:
#             shear = np.random.uniform(shears[0], shears[1])
#         else:
#             shear = 0.0

#         return angle, translations, scale, shear

#     def __call__(self, img1, img2, transform_param1, transform_param2):
#         ret1 = self.get_params(self.degrees, self.translate, self.scale, self.shear, img1.size)
#         ret2 = self.get_params(self.degrees, self.translate, self.scale, self.shear, img2.size)
#         _ret1 = (-ret1[0],(-ret1[1][0],-ret1[1][1]), 1/ret1[2], -ret1[3])
#         _ret2 = (-ret2[0],(-ret2[1][0],-ret2[1][1]), 1/ret2[2], -ret2[3])
#         transform_param1[1:] = [_ret1[0], _ret1[1][0], _ret1[1][1], _ret1[2], _ret1[3]]
#         transform_param2[1:] = [_ret2[0], _ret2[1][0], _ret2[1][1], _ret2[2], _ret2[3]]

#         return affine(img1, *ret1, resample=self.resample, fillcolor=self.fillcolor_rgb), affine(img2, *ret2, resample=self.resample, fillcolor=self.fillcolor_rgb), transform_param1, transform_param2, affine(lbl1, *ret1, resample=self.resample, fillcolor=self.fillcolor_rgb), affine(lbl2, *ret2, resample=self.resample, fillcolor=self.fillcolor_rgb)

# class Tgt_Random_Sync_Affine():

#     def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor_rgb=(0,0,0), fillcolor_gray=255):
#         if isinstance(degrees, numbers.Number):
#             if degrees < 0:
#                 raise ValueError("If degrees is a single number, it must be positive.")
#             self.degrees = (-degrees, degrees)
#         else:
#             assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
#                 "degrees should be a list or tuple and it must be of length 2."
#             self.degrees = degrees

#         if translate is not None:
#             assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
#                 "translate should be a list or tuple and it must be of length 2."
#             for t in translate:
#                 if not (0.0 <= t <= 1.0):
#                     raise ValueError("translation values should be between 0 and 1")
#         self.translate = translate

#         if scale is not None:
#             assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
#                 "scale should be a list or tuple and it must be of length 2."
#             for s in scale:
#                 if s <= 0:
#                     raise ValueError("scale values should be positive")
#         self.scale = scale

#         if shear is not None:
#             if isinstance(shear, numbers.Number):
#                 if shear < 0:
#                     raise ValueError("If shear is a single number, it must be positive.")
#                 self.shear = (-shear, shear)
#             else:
#                 assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
#                     "shear should be a list or tuple and it must be of length 2."
#                 self.shear = shear
#         else:
#             self.shear = shear

#         self.resample = resample
#         self.fillcolor_rgb = fillcolor_rgb
#         self.fillcolor_gray = fillcolor_gray

#     @staticmethod
#     def get_params(degrees, translate, scale_ranges, shears, img_size):
#         """Get parameters for affine transformation

#         Returns:
#             sequence: params to be passed to the affine transformation
#         """
#         angle = np.random.uniform(degrees[0], degrees[1])
#         if translate is not None:
#             max_dx = translate[0] * img_size[0]
#             max_dy = translate[1] * img_size[1]
#             translations = (np.round(np.random.uniform(-max_dx, max_dx)),
#                             np.round(np.random.uniform(-max_dy, max_dy)))
#         else:
#             translations = (0, 0)

#         if scale_ranges is not None:
#             scale = np.random.uniform(scale_ranges[0], scale_ranges[1])
#         else:
#             scale = 1.0

#         if shears is not None:
#             shear = np.random.uniform(shears[0], shears[1])
#         else:
#             shear = 0.0

#         return angle, translations, scale, shear

#     def __call__(self, img1, img2, transform_param1, transform_param2):
#         ret1 = self.get_params(self.degrees, self.translate, self.scale, self.shear, img1.size)
#         _ret1 = (-ret1[0],(-ret1[1][0],-ret1[1][1]), 1/ret1[2], -ret1[3])
#         transform_param1[1:] = [_ret1[0], _ret1[1][0], _ret1[1][1], _ret1[2], _ret1[3]]

#         return affine(img1, *ret1, resample=self.resample, fillcolor=self.fillcolor_rgb), affine(img2, *ret1, resample=self.resample, fillcolor=self.fillcolor_rgb), transform_param1, transform_param1

# class Tgt_Random_Sync_Affine_with_label():

#     def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor_rgb=(0,0,0), fillcolor_gray=255):
#         if isinstance(degrees, numbers.Number):
#             if degrees < 0:
#                 raise ValueError("If degrees is a single number, it must be positive.")
#             self.degrees = (-degrees, degrees)
#         else:
#             assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
#                 "degrees should be a list or tuple and it must be of length 2."
#             self.degrees = degrees

#         if translate is not None:
#             assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
#                 "translate should be a list or tuple and it must be of length 2."
#             for t in translate:
#                 if not (0.0 <= t <= 1.0):
#                     raise ValueError("translation values should be between 0 and 1")
#         self.translate = translate

#         if scale is not None:
#             assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
#                 "scale should be a list or tuple and it must be of length 2."
#             for s in scale:
#                 if s <= 0:
#                     raise ValueError("scale values should be positive")
#         self.scale = scale

#         if shear is not None:
#             if isinstance(shear, numbers.Number):
#                 if shear < 0:
#                     raise ValueError("If shear is a single number, it must be positive.")
#                 self.shear = (-shear, shear)
#             else:
#                 assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
#                     "shear should be a list or tuple and it must be of length 2."
#                 self.shear = shear
#         else:
#             self.shear = shear

#         self.resample = resample
#         self.fillcolor_rgb = fillcolor_rgb
#         self.fillcolor_gray = fillcolor_gray

#     @staticmethod
#     def get_params(degrees, translate, scale_ranges, shears, img_size):
#         """Get parameters for affine transformation

#         Returns:
#             sequence: params to be passed to the affine transformation
#         """
#         angle = np.random.uniform(degrees[0], degrees[1])
#         if translate is not None:
#             max_dx = translate[0] * img_size[0]
#             max_dy = translate[1] * img_size[1]
#             translations = (np.round(np.random.uniform(-max_dx, max_dx)),
#                             np.round(np.random.uniform(-max_dy, max_dy)))
#         else:
#             translations = (0, 0)

#         if scale_ranges is not None:
#             scale = np.random.uniform(scale_ranges[0], scale_ranges[1])
#         else:
#             scale = 1.0

#         if shears is not None:
#             shear = np.random.uniform(shears[0], shears[1])
#         else:
#             shear = 0.0

#         return angle, translations, scale, shear

#     def __call__(self, img1, img2, transform_param1, transform_param2, lbl1, lbl2):
#         ret1 = self.get_params(self.degrees, self.translate, self.scale, self.shear, img1.size)
#         _ret1 = (-ret1[0],(-ret1[1][0],-ret1[1][1]), 1/ret1[2], -ret1[3])
#         transform_param1[1:] = [_ret1[0], _ret1[1][0], _ret1[1][1], _ret1[2], _ret1[3]]

#         return affine(img1, *ret1, resample=self.resample, fillcolor=self.fillcolor_rgb), affine(img2, *ret1, resample=self.resample, fillcolor=self.fillcolor_rgb), transform_param1, transform_param1, affine(lbl1, *ret1, resample=self.resample, fillcolor=self.fillcolor_gray), affine(lbl2, *ret1, resample=self.resample, fillcolor=self.fillcolor_gray)

class Tgt_jitter():

    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0., brightness_2=None, contrast_2=None, saturation_2=None, hue_2=None):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        if brightness_2 is not None:
            self.brightness_2 = brightness_2
        else:
            self.brightness_2 = brightness

        if contrast_2 is not None:
            self.contrast_2 = contrast_2
        else:
            self.contrast_2 = contrast
            
        if saturation_2 is not None:
            self.saturation_2 = saturation_2
        else:
            self.saturation_2 = saturation

        if hue_2 is not None:
            self.hue_2 = hue_2
        else:
            self.hue_2 = hue

    def __call__(self, img1, img2, lbl1, lbl2, transform_param1, transform_param2):
        original_jitter = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        original_jitter2 = transforms.ColorJitter(self.brightness_2, self.contrast_2, self.saturation_2, self.hue_2)
        return original_jitter(img1), original_jitter2(img2), lbl1, lbl2, transform_param1, transform_param2

class Tgt_Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms
        self.transform_param1 = np.array([0 for i in range(6)]) # protocol: an array of a length of 6: [flip_flag, rot_angle, translations_x, translations_y, scale, shear_angle]
        self.transform_param2 = np.array([0 for i in range(6)]) # protocol: an array of a length of 6: [flip_flag, rot_angle, translations_x, translations_y, scale, shear_angle]

    def __call__(self, img1, img2, lbl1, lbl2):
        for t in self.transforms:
            img1, img2, lbl1, lbl2, self.transform_param1, self.transform_param2 = t(img1, img2, lbl1, lbl2, self.transform_param1, self.transform_param2)
        return img1, img2, lbl1, lbl2, self.transform_param1, self.transform_param2

# class Tgt_Compose_with_label(object):

#     def __init__(self, transforms):
#         self.transforms = transforms
#         self.transform_param1 = np.array([0 for i in range(6)]) # protocol: an array of a length of 6: [flip_flag, rot_angle, translations_x, translations_y, scale, shear_angle]
#         self.transform_param2 = np.array([0 for i in range(6)]) # protocol: an array of a length of 6: [flip_flag, rot_angle, translations_x, translations_y, scale, shear_angle]

#     def __call__(self, img1, img2, lbl1, lbl2):
#         for t in self.transforms:
#             img1, img2, self.transform_param1, self.transform_param2, lbl1, lbl2 = t(img1, img2, self.transform_param1, self.transform_param2, lbl1, lbl2)
#         return img1, img2, self.transform_param1, self.transform_param2, lbl1, lbl2

# class Tgt_inv_Horizontal_Flip:

#     def __init__(self, flip_param1, flip_param2):
#         self.flip_param1 = flip_param1
#         self.flip_param2 = flip_param2

#     def __call__(self, img1, img2):
#         if self.flip_param1:
#             img1 = F.hflip(img1)
#         if self.flip_param2:
#             img2 = F.hflip(img2)

#         return img1, img2

# class Tgt_inv_RandomAffine():

#     def __init__(self, affine_param1, affine_param2):
#         self.affine_param1 = affine_param1
#         self.affine_param2 = affine_param2

#     def __call__(self, img1, img2):
#         if img1.mode == "RGB":
#             self.fillcolor = (0,0,0)
#         else:
#             self.fillcolor = 255

#         img1 = affine(img1, *self.affine_param1, fillcolor=self.fillcolor)
#         img2 = affine(img2, *self.affine_param2, fillcolor=self.fillcolor)

#         return img1, img2

# class Tgt_inv_toPIL():
    
#     def __init__(self, mode=None):
#         self.mode = mode

#     def __call__(self, img1, img2):
#         return F.to_pil_image(img1, self.mode), F.to_pil_image(img2, self.mode)

# class Tgt_inv_totensor():

#     def __init__(self):
#         pass

#     def __call__(self, img1, img2):
#         return F.to_tensor(img1), F.to_tensor(img2)

# class Tgt_inv_Compose(object):

#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, img1, img2):
#         for t in self.transforms:
#             img1, img2 = t(img1, img2)
#         return img1, img2


class Src_Horizontal_Flip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if np.random.random() < self.p:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl

class Src_RandomAffine():

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor_rgb=(0,0,0), fillcolor_gray=255):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor_rgb = fillcolor_rgb
        self.fillcolor_gray = fillcolor_gray

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = np.random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                            np.round(np.random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = np.random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = np.random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img, lbl):
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        return affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor_rgb), affine(lbl, *ret, resample=self.resample, fillcolor=self.fillcolor_gray)

class Src_jitter():

    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0.):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img, lbl):
        original_jitter = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        return original_jitter(img), lbl

class Src_Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl):
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl
