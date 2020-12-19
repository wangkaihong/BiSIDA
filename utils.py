import numpy as np
import copy
import six
import skimage
import skimage.color
import skimage.transform
import numbers
from distutils.version import LooseVersion
import math

cp_color_map = [
    (128, 64, 128), 
    (244, 35, 232), 
    (70, 70, 70),
    (102, 102, 156), 
    (190, 153, 153), 
    (153, 153, 153),
    (250, 170, 30), 
    (220, 220,  0), 
    (107, 142, 35), 
    (152, 251, 152), 
    (70,130,180),
    (220, 20, 60), 
    (255, 0, 0), 
    (0, 0, 142), 
    (0, 0, 70), 
    (0, 60, 100), 
    (0, 80, 100), 
    (0, 0, 230), 
    (119, 11, 32),
    (0, 0, 0)
]

def rgb2bgr(color):
    return color[:, ::-1]

def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a + shear_y)*scale    -sin(a + shear_x)*scale     0]
    #                              [ sin(a + shear_y)*scale    cos(a + shear_x)*scale     0]
    #                              [     0                  0          1]
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    angle = math.radians(angle)
    if isinstance(shear, (tuple, list)) and len(shear) == 2:
        shear = [math.radians(s) for s in shear]
    elif isinstance(shear, numbers.Number):
        shear = math.radians(shear)
        shear = [shear, 0]
    else:
        raise ValueError(
            "Shear should be a single value or a tuple/list containing " +
            "two values. Got {}".format(shear))
    scale = 1.0 / scale

    # Inverted rotation matrix with scale and shear
    d = math.cos(angle + shear[0]) * math.cos(angle + shear[1]) + \
        math.sin(angle + shear[0]) * math.sin(angle + shear[1])
    matrix = [
        math.cos(angle + shear[0]), math.sin(angle + shear[0]), 0,
        -math.sin(angle + shear[1]), math.cos(angle + shear[1]), 0
    ]
    matrix = [scale / d * m for m in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
    matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]
    return matrix

def _fast_cate_hist(cates, n_class):
    hist = np.bincount(cates, minlength=n_class)
    return hist

def _fast_pred_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class) # mask or not
    hist = np.bincount(label_pred[mask], minlength=n_class)
    return hist

def class_dist_stat(label_trues, label_preds, n_class):
    hists = np.zeros((n_class,))
    for lt, lp in zip(label_trues, label_preds):
        hists += _fast_pred_hist(lt.flatten(), lp.flatten(), n_class)
    hists = hists / len(label_trues)
    hists = hists / sum(hists)
    return hists

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, iu

def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)

def centerize(src, dst_shape, margin_color=None):
    """Centerize image for specified image size
    @param src: image to centerize
    @param dst_shape: image shape (height, width) or (height, width, channel)
    """
    if src.shape[:2] == dst_shape[:2]:
        return src
    centerized = np.zeros(dst_shape, dtype=src.dtype)
    if margin_color:
        centerized[:, :] = margin_color
    pad_vertical, pad_horizontal = 0, 0
    h, w = src.shape[:2]
    dst_h, dst_w = dst_shape[:2]
    if h < dst_h:
        pad_vertical = (dst_h - h) // 2
    if w < dst_w:
        pad_horizontal = (dst_w - w) // 2
    centerized[pad_vertical:pad_vertical + h,
               pad_horizontal:pad_horizontal + w] = src
    return centerized

def _tile_images(imgs, tile_shape, concatenated_image):
    """Concatenate images whose sizes are same.
    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param concatenated_image: returned image.
        if it is None, new image will be created.
    """
    y_num, x_num = tile_shape
    one_width = imgs[0].shape[1]
    one_height = imgs[0].shape[0]
    if concatenated_image is None:
        if len(imgs[0].shape) == 3:
            n_channels = imgs[0].shape[2]
            assert all(im.shape[2] == n_channels for im in imgs)
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num, n_channels),
                dtype=np.uint8,
            )
        else:
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num), dtype=np.uint8)
    for y in six.moves.range(y_num):
        for x in six.moves.range(x_num):
            i = x + y * x_num
            if i >= len(imgs):
                pass
            else:
                concatenated_image[y * one_height:(y + 1) * one_height,
                                   x * one_width:(x + 1) * one_width] = imgs[i]
    return concatenated_image

def label_colormap(N=256):
    cmap = np.zeros((N, 3))
    for i in six.moves.range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in six.moves.range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap

def get_tile_image(imgs, tile_shape=None, result_img=None, margin_color=None):
    """Concatenate images whose sizes are different.
    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param result_img: numpy array to put result image
    """
    def resize(*args, **kwargs):
        # anti_aliasing arg cannot be passed to skimage<0.14
        # use LooseVersion to allow 0.14dev.
        if LooseVersion(skimage.__version__) < LooseVersion('0.14'):
            kwargs.pop('anti_aliasing', None)
        return skimage.transform.resize(*args, **kwargs)

    def get_tile_shape(img_num):
        x_num = 0
        y_num = int(math.sqrt(img_num))
        while x_num * y_num < img_num:
            x_num += 1
        return y_num, x_num

    if tile_shape is None:
        tile_shape = get_tile_shape(len(imgs))

    # get max tile size to which each image should be resized
    max_height, max_width = np.inf, np.inf
    for img in imgs:
        max_height = min([max_height, img.shape[0]])
        max_width = min([max_width, img.shape[1]])

    # resize and concatenate images
    for i, img in enumerate(imgs):
        h, w = img.shape[:2]
        dtype = img.dtype
        h_scale, w_scale = max_height / h, max_width / w
        scale = min([h_scale, w_scale])
        h, w = int(scale * h), int(scale * w)
        img = resize(
            image=img,
            output_shape=(h, w),
            mode='reflect',
            preserve_range=True,
            anti_aliasing=True,
        ).astype(dtype)
        if len(img.shape) == 3:
            img = centerize(img, (max_height, max_width, 3), margin_color)
        else:
            img = centerize(img, (max_height, max_width), margin_color)
        imgs[i] = img
    return _tile_images(imgs, tile_shape, result_img)

def label2rgb(lbl, img=None, label_names=None, n_labels=None, cmap=None,
              alpha=0.5, thresh_suppress=0):
    if label_names is None:
        if n_labels is None:
            n_labels = lbl.max() + 1  # +1 for bg_label 0
    else:
        if n_labels is None:
            n_labels = len(label_names)
        else:
            assert n_labels == len(label_names)
    if cmap is not None:
        cmap = rgb2bgr(np.array(cmap))
    else:
        cmap = label_colormap(n_labels)
        cmap = (cmap * 255).astype(np.uint8)

    lbl_viz = cmap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled

    if img is not None:
        img_gray = skimage.color.rgb2gray(img)
        img_gray = skimage.color.gray2rgb(img_gray)
        img_gray *= 255
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)

    if label_names is None:
        return lbl_viz

    # cv2 is required only if label_names is not None
    import cv2
    if cv2 is None:
        warnings.warn('label2rgb with label_names requires OpenCV (cv2), '
                      'so ignoring label_names values.')
        return lbl_viz

    np.random.seed(1234)
    for label in np.unique(lbl):
        if label == -1:
            continue  # unlabeled

        mask = lbl == label
        if 1. * mask.sum() / mask.size < thresh_suppress:
            continue
        mask = (mask * 255).astype(np.uint8)
        y, x = scipy.ndimage.center_of_mass(mask)
        y, x = map(int, [y, x])

        if lbl[y, x] != label:
            Y, X = np.where(mask)
            point_index = np.random.randint(0, len(Y))
            y, x = Y[point_index], X[point_index]

        text = label_names[label]
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size, baseline = cv2.getTextSize(
            text, font_face, font_scale, thickness)

        def get_text_color(color):
            if color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114 > 170:
                return (0, 0, 0)
            return (255, 255, 255)

        color = get_text_color(lbl_viz[y, x])
        cv2.putText(lbl_viz, text,
                    (x - text_size[0] // 2, y),
                    font_face, font_scale, color, thickness)
    return lbl_viz

def visualize_segmentation(**kwargs):
    """Visualize segmentation.
    Parameters
    ----------
    img: ndarray
        Input image to predict label.
    lbl_true: ndarray
        Ground truth of the label.
    lbl_pred: ndarray
        Label predicted.
    n_class: int
        Number of classes.
    label_names: dict or list
        Names of each label value.
        Key or index is label_value and value is its name.
    Returns
    -------
    img_array: ndarray
        Visualized image.
    """
    img = kwargs.pop('img', None)
    lbl_true = kwargs.pop('lbl_true', None)
    lbl_pred = kwargs.pop('lbl_pred', None)
    n_class = kwargs.pop('n_class', None)
    label_names = kwargs.pop('label_names', None)
    if kwargs:
        raise RuntimeError(
            'Unexpected keys in kwargs: {}'.format(kwargs.keys()))

    if lbl_true is None and lbl_pred is None:
        raise ValueError('lbl_true or lbl_pred must be not None.')

    lbl_true = copy.deepcopy(lbl_true)
    lbl_pred = copy.deepcopy(lbl_pred)

    mask_unlabeled = None
    viz_unlabeled = None
    if lbl_true is not None:
        mask_unlabeled = lbl_true == -1
        lbl_true[mask_unlabeled] = 0
        viz_unlabeled = (
            np.random.random((lbl_true.shape[0], lbl_true.shape[1], 3)) * 255
        ).astype(np.uint8)
        if lbl_pred is not None:
            lbl_pred[mask_unlabeled] = 0

    vizs = []

    if lbl_true is not None:
        viz_trues = [
            img,
            label2rgb(lbl_true, label_names=label_names, n_labels=n_class, cmap=cp_color_map),
            label2rgb(lbl_true, img, label_names=label_names,
                      n_labels=n_class, cmap=cp_color_map),
        ]
        viz_trues[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        viz_trues[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        vizs.append(get_tile_image(viz_trues, (1, 3)))

    if lbl_pred is not None:
        viz_preds = [
            img,
            label2rgb(lbl_pred, label_names=label_names, n_labels=n_class, cmap=cp_color_map),
            label2rgb(lbl_pred, img, label_names=label_names,
                      n_labels=n_class, cmap=cp_color_map),
        ]
        if mask_unlabeled is not None and viz_unlabeled is not None:
            viz_preds[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
            viz_preds[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        vizs.append(get_tile_image(viz_preds, (1, 3)))

    if len(vizs) == 1:
        return vizs[0]
    elif len(vizs) == 2:
        return get_tile_image(vizs, (2, 1))
    else:
        raise RuntimeError

def vis_heat(img):
    if np.max(img) > 0:
        img = img / np.max(img) * 255.
    img = np.concatenate((np.expand_dims(img, 2), np.zeros((*(img.shape), 2))), axis=2)
    img = img.astype(np.uint8)
    return img

def visualize_segmentation_aug(**kwargs):
    
    src_img = kwargs.pop('src_img', None) #
    tgt_img_1 = kwargs.pop('tgt_img_1', None) #
    tgt_img_2 = kwargs.pop('tgt_img_2', None) #
    src_pred = kwargs.pop('src_pred', None) #
    src_lbl = kwargs.pop('src_lbl', None) #
    lbl_true_1 = kwargs.pop('lbl_true_1', None) #
    lbl_true_2 = kwargs.pop('lbl_true_2', None) #
    lbl_pred_stu = kwargs.pop('lbl_pred_stu', None) #
    lbl_pred_tea = kwargs.pop('lbl_pred_tea', None) #
    n_class = kwargs.pop('n_class', None)
    label_names = kwargs.pop('label_names', None)
    aug_loss = kwargs.pop('aug_loss', None)
    aug_loss_dist = kwargs.pop('aug_loss_dist', None)
    masked_aug_loss_dist = kwargs.pop('masked_aug_loss_dist', None)
    unsup_mask = kwargs.pop('unsup_mask', None)
    if kwargs:
        raise RuntimeError(
            'Unexpected keys in kwargs: {}'.format(kwargs.keys()))

    if lbl_true_1 is None and lbl_true_2 is None and lbl_pred_stu is None and lbl_pred_tea is None:
        raise ValueError('lbl_true or lbl_pred_stu or lbl_pred_tea must be not None.')

    lbl_true_1 = copy.deepcopy(lbl_true_1)
    lbl_true_2 = copy.deepcopy(lbl_true_2)
    lbl_pred_stu = copy.deepcopy(lbl_pred_stu)
    lbl_pred_tea = copy.deepcopy(lbl_pred_tea)

    mask_unlabeled = None
    viz_unlabeled = None
    if lbl_true_1 is not None:
        mask_unlabeled = lbl_true_1 == -1
        lbl_true_1[mask_unlabeled] = 0
        viz_unlabeled = (
            np.random.random((lbl_true_1.shape[0], lbl_true_1.shape[1], 3)) * 255
        ).astype(np.uint8)
        if lbl_pred_stu is not None:
            lbl_pred_stu[mask_unlabeled] = 0
        if lbl_pred_tea is not None:
            lbl_pred_tea[mask_unlabeled] = 0

    vizs = []

    aug_loss_dist_heatmap = vis_heat(aug_loss_dist)
    masked_aug_loss_dist_heatmap = vis_heat(masked_aug_loss_dist)
    aug_loss_heatmap = vis_heat(aug_loss)
    unsup_mask = vis_heat(unsup_mask)

    viz_src = [
        src_img,
        label2rgb(src_lbl, label_names=label_names, n_labels=n_class, cmap=cp_color_map),
        label2rgb(src_lbl, src_img, label_names=label_names,
                    n_labels=n_class, cmap=cp_color_map),
        label2rgb(src_pred, label_names=label_names, n_labels=n_class, cmap=cp_color_map),
        label2rgb(src_pred, src_img, label_names=label_names,
                    n_labels=n_class, cmap=cp_color_map)
    ]
    if mask_unlabeled is not None and viz_unlabeled is not None:
        viz_src[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        viz_src[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
    vizs.append(get_tile_image(viz_src, (1, 5)))

    viz_tgt_1 = [
        tgt_img_1,
        label2rgb(lbl_true_1, label_names=label_names, n_labels=n_class, cmap=cp_color_map),
        label2rgb(lbl_true_1, tgt_img_1, label_names=label_names,
                    n_labels=n_class, cmap=cp_color_map),
        label2rgb(lbl_pred_stu, label_names=label_names, n_labels=n_class, cmap=cp_color_map),
        label2rgb(lbl_pred_stu, tgt_img_1, label_names=label_names,
                    n_labels=n_class, cmap=cp_color_map)
    ]
    if mask_unlabeled is not None and viz_unlabeled is not None:
        viz_tgt_1[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        viz_tgt_1[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
    vizs.append(get_tile_image(viz_tgt_1, (1, 5)))

    viz_tgt_2 = [
        tgt_img_2,
        label2rgb(lbl_true_2, label_names=label_names, n_labels=n_class, cmap=cp_color_map),
        label2rgb(lbl_true_2, tgt_img_2, label_names=label_names,
                    n_labels=n_class, cmap=cp_color_map),
        label2rgb(lbl_pred_tea, label_names=label_names, n_labels=n_class, cmap=cp_color_map),
        label2rgb(lbl_pred_tea, tgt_img_2, label_names=label_names,
                    n_labels=n_class, cmap=cp_color_map)
    ]
    if mask_unlabeled is not None and viz_unlabeled is not None:
        viz_tgt_2[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        viz_tgt_2[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
    vizs.append(get_tile_image(viz_tgt_2, (1, 5)))

    viz_maps = [
        aug_loss_heatmap,
        aug_loss_dist_heatmap,
        unsup_mask,
        masked_aug_loss_dist_heatmap,
        masked_aug_loss_dist_heatmap,
    ]
    vizs.append(get_tile_image(viz_maps, (1, 5)))

    if len(vizs) == 1:
        return vizs[0]
    else:
        return get_tile_image(vizs, (4, 1))

# def visualize_segmentation_aug_mix(**kwargs):
    
#     src_img = kwargs.pop('src_img', None) #
#     tgt_img_1 = kwargs.pop('tgt_img_1', None) #
#     tgt_img_2 = kwargs.pop('tgt_img_2', None) #
#     tgt_img_merge = kwargs.pop('tgt_img_merge', None) #
#     src_pred = kwargs.pop('src_pred', None) #
#     src_lbl = kwargs.pop('src_lbl', None) #
#     lbl_true_1 = kwargs.pop('lbl_true_1', None) #
#     lbl_true_2 = kwargs.pop('lbl_true_2', None) #
#     lbl_pred_merge = kwargs.pop('lbl_pred_merge', None) #
#     lbl_merge_pred = kwargs.pop('lbl_merge_pred', None) #
#     lbl_pred_stu = kwargs.pop('lbl_pred_stu', None) #
#     lbl_pred_tea = kwargs.pop('lbl_pred_tea', None) #
#     lbl_true_merge = kwargs.pop('lbl_true_merge', None) #
#     n_class = kwargs.pop('n_class', None)
#     label_names = kwargs.pop('label_names', None)
#     aug_loss = kwargs.pop('aug_loss', None)
#     aug_loss_dist = kwargs.pop('aug_loss_dist', None)
#     masked_aug_loss_dist = kwargs.pop('masked_aug_loss_dist', None)
#     unsup_mask = kwargs.pop('unsup_mask', None)
#     if kwargs:
#         raise RuntimeError(
#             'Unexpected keys in kwargs: {}'.format(kwargs.keys()))

#     if lbl_true_1 is None and lbl_true_2 is None and lbl_pred_stu is None and lbl_pred_tea is None:
#         raise ValueError('lbl_true or lbl_pred_stu or lbl_pred_tea must be not None.')

#     lbl_true_1 = copy.deepcopy(lbl_true_1)
#     lbl_true_2 = copy.deepcopy(lbl_true_2)
#     lbl_pred_stu = copy.deepcopy(lbl_pred_stu)
#     lbl_pred_tea = copy.deepcopy(lbl_pred_tea)
#     lbl_true_merge = copy.deepcopy(lbl_true_merge)

#     mask_unlabeled = None
#     viz_unlabeled = None
#     if lbl_true_1 is not None:
#         mask_unlabeled = lbl_true_1 == -1
#         lbl_true_1[mask_unlabeled] = 0
#         viz_unlabeled = (
#             np.random.random((lbl_true_1.shape[0], lbl_true_1.shape[1], 3)) * 255
#         ).astype(np.uint8)
#         if lbl_pred_stu is not None:
#             lbl_pred_stu[mask_unlabeled] = 0
#         if lbl_pred_tea is not None:
#             lbl_pred_tea[mask_unlabeled] = 0
#         if lbl_true_merge is not None:
#             lbl_true_merge[mask_unlabeled] = 0

#     vizs = []

#     aug_loss_dist_heatmap = vis_heat(aug_loss_dist)
#     masked_aug_loss_dist_heatmap = vis_heat(masked_aug_loss_dist)
#     aug_loss_heatmap = vis_heat(aug_loss)
#     unsup_mask = vis_heat(unsup_mask)

#     viz_src = [
#         src_img,
#         label2rgb(src_lbl, label_names=label_names, n_labels=n_class),
#         label2rgb(src_lbl, src_img, label_names=label_names,
#                     n_labels=n_class),
#         label2rgb(src_pred, label_names=label_names, n_labels=n_class),
#         label2rgb(src_pred, src_img, label_names=label_names,
#                     n_labels=n_class)
#     ]
#     if mask_unlabeled is not None and viz_unlabeled is not None:
#         viz_src[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
#         viz_src[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
#     vizs.append(get_tile_image(viz_src, (1, 5)))

#     viz_tgt_1 = [
#         tgt_img_1,
#         label2rgb(lbl_true_1, label_names=label_names, n_labels=n_class),
#         label2rgb(lbl_true_1, tgt_img_1, label_names=label_names,
#                     n_labels=n_class),
#         label2rgb(lbl_pred_stu, label_names=label_names, n_labels=n_class),
#         label2rgb(lbl_pred_stu, tgt_img_1, label_names=label_names,
#                     n_labels=n_class)
#     ]
#     if mask_unlabeled is not None and viz_unlabeled is not None:
#         viz_tgt_1[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
#         viz_tgt_1[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
#     vizs.append(get_tile_image(viz_tgt_1, (1, 5)))

#     viz_tgt_2 = [
#         tgt_img_2,
#         label2rgb(lbl_true_2, label_names=label_names, n_labels=n_class),
#         label2rgb(lbl_true_2, tgt_img_2, label_names=label_names,
#                     n_labels=n_class),
#         label2rgb(lbl_pred_tea, label_names=label_names, n_labels=n_class),
#         label2rgb(lbl_pred_tea, tgt_img_2, label_names=label_names,
#                     n_labels=n_class)
#     ]
#     if mask_unlabeled is not None and viz_unlabeled is not None:
#         viz_tgt_2[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
#         viz_tgt_2[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
#     vizs.append(get_tile_image(viz_tgt_2, (1, 5)))

#     viz_tgt_3 = [
#         tgt_img_merge,
#         label2rgb(lbl_true_merge, label_names=label_names, n_labels=n_class),
#         label2rgb(lbl_true_merge, tgt_img_merge, label_names=label_names,
#                     n_labels=n_class),
#         label2rgb(lbl_pred_merge, label_names=label_names, n_labels=n_class),
#         label2rgb(lbl_pred_merge, tgt_img_merge, label_names=label_names,
#                     n_labels=n_class),
#         label2rgb(lbl_merge_pred, label_names=label_names, n_labels=n_class),
#         label2rgb(lbl_merge_pred, tgt_img_merge, label_names=label_names,
#                     n_labels=n_class)
#     ]
#     if mask_unlabeled is not None and viz_unlabeled is not None:
#         viz_tgt_3[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
#         viz_tgt_3[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
#     vizs.append(get_tile_image(viz_tgt_3, (1, 7)))

#     viz_maps = [
#         aug_loss_heatmap,
#         aug_loss_dist_heatmap,
#         unsup_mask,
#         masked_aug_loss_dist_heatmap,
#         masked_aug_loss_dist_heatmap,
#     ]
#     vizs.append(get_tile_image(viz_maps, (1, 5)))

#     if len(vizs) == 1:
#         return vizs[0]
#     else:
#         return get_tile_image(vizs, (5, 1))

def visualize_segmentation_for_presentation(**kwargs):
    """Visualize segmentation.
    Parameters
    ----------
    img: ndarray
        Input image to predict label.
    lbl_true: ndarray
        Ground truth of the label.
    lbl_pred: ndarray
        Label predicted.
    n_class: int
        Number of classes.
    label_names: dict or list
        Names of each label value.
        Key or index is label_value and value is its name.
    Returns
    -------
    img_array: ndarray
        Visualized image.
    """
    img = kwargs.pop('img', None)
    lbl_true = kwargs.pop('lbl_true', None)
    lbl_pred1 = kwargs.pop('lbl_pred1', None)
    lbl_pred2 = kwargs.pop('lbl_pred2', None)
    n_class = kwargs.pop('n_class', None)
    label_names = kwargs.pop('label_names', None)
    if kwargs:
        raise RuntimeError(
            'Unexpected keys in kwargs: {}'.format(kwargs.keys()))

    if lbl_true is None and lbl_pred is None:
        raise ValueError('lbl_true or lbl_pred must be not None.')

    lbl_true = copy.deepcopy(lbl_true)
    lbl_pred1 = copy.deepcopy(lbl_pred1)
    lbl_pred2 = copy.deepcopy(lbl_pred2)

    mask_unlabeled = None
    viz_unlabeled = None
    if lbl_true is not None:
        mask_unlabeled = lbl_true == -1
        lbl_true[mask_unlabeled] = 0
        viz_unlabeled = (
            np.random.random((lbl_true.shape[0], lbl_true.shape[1], 3)) * 255
        ).astype(np.uint8)
        if lbl_pred1 is not None:
            lbl_pred1[mask_unlabeled] = 0
        if lbl_pred2 is not None:
            lbl_pred2[mask_unlabeled] = 0

    vizs = []

    if lbl_pred1 is not None and lbl_pred2 is not None and lbl_true is not None:
        viz_preds = [
            img,
            label2rgb(lbl_true, label_names=label_names, n_labels=n_class, cmap=cp_color_map),
            label2rgb(lbl_pred2, label_names=label_names, n_labels=n_class, cmap=cp_color_map),
            label2rgb(lbl_pred1, label_names=label_names, n_labels=n_class, cmap=cp_color_map),
        ]
        if mask_unlabeled is not None and viz_unlabeled is not None:
            viz_preds[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
            viz_preds[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
            viz_preds[3][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        vizs.append(get_tile_image(viz_preds, (1, 4)))

    return viz_preds

def visualize_segmentation_for_presentation2(**kwargs):
    """Visualize segmentation.
    Parameters
    ----------
    img: ndarray
        Input image to predict label.
    lbl_true: ndarray
        Ground truth of the label.
    lbl_pred: ndarray
        Label predicted.
    n_class: int
        Number of classes.
    label_names: dict or list
        Names of each label value.
        Key or index is label_value and value is its name.
    Returns
    -------
    img_array: ndarray
        Visualized image.
    """
    img = kwargs.pop('img', None)
    lbl_true = kwargs.pop('lbl_true', None)
    lbl_pred1 = kwargs.pop('lbl_pred1', None)
    lbl_pred2 = kwargs.pop('lbl_pred2', None)
    n_class = kwargs.pop('n_class', None)
    label_names = kwargs.pop('label_names', None)
    if kwargs:
        raise RuntimeError(
            'Unexpected keys in kwargs: {}'.format(kwargs.keys()))

    if lbl_true is None and lbl_pred is None:
        raise ValueError('lbl_true or lbl_pred must be not None.')

    lbl_true = copy.deepcopy(lbl_true)
    lbl_pred1 = copy.deepcopy(lbl_pred1)
    lbl_pred2 = copy.deepcopy(lbl_pred2)

    mask_unlabeled = None
    viz_unlabeled = None
    if lbl_true is not None:
        mask_unlabeled = lbl_true == -1
        lbl_true[mask_unlabeled] = 0
        viz_unlabeled = (
            np.random.random((lbl_true.shape[0], lbl_true.shape[1], 3)) * 255
        ).astype(np.uint8)
        if lbl_pred1 is not None:
            lbl_pred1[mask_unlabeled] = 0
        if lbl_pred2 is not None:
            lbl_pred2[mask_unlabeled] = 0

    vizs = []

    if lbl_pred1 is not None and lbl_pred2 is not None and lbl_true is not None:
        viz_preds = [
            img,
            label2rgb(lbl_true, label_names=label_names, n_labels=n_class, cmap=cp_color_map),
            label2rgb(lbl_pred2, label_names=label_names, n_labels=n_class, cmap=cp_color_map),
            label2rgb(lbl_pred1, label_names=label_names, n_labels=n_class, cmap=cp_color_map),
        ]
        if mask_unlabeled is not None and viz_unlabeled is not None:
            viz_preds[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
            viz_preds[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
            viz_preds[3][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        vizs.append(get_tile_image(viz_preds, (1, 4)))

    return vizs[0]
