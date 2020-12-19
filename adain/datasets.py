import collections
import os.path as osp
import os
import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import cv2
from torchvision import transforms

class Synthia(data.Dataset):

    img_mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)

    class_names = np.array([
        'void',
        'sky',
        'Building',
        'Road',
        'Sidewalk',
        'Fence',
        'Vegetation',
        'Pole',
        'Car',
        'Traffic sign',
        'Pedestrian',
        'Bicycle',
        'Motorcycle',
        'Parking-slot',
        'Road-work',
        'Traffic light',
        'Terrain',
        'Rider',
        'Truck',
        'Bus',
        'Train',
        'Wall',
        'Lanemarking',
    ])

    # synthia_id_to_trainid = {
    #     3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
    #     15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
    #     8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18
    # }
    synthia_id_to_trainid = {
        3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
        15: 6, 9: 7, 6: 8, 1: 9, 10: 10, 17: 11,
        8: 12, 19: 13, 12: 14, 11: 15
    }

    rep_class_names = np.array([
        'road',
        'sidewalk',
        'building',
        'wall',
        'fence',
        'pole',
        'traffic light',
        'traffic sign',
        'vegetation',
        # 'terrain',
        'sky',
        'person',
        'rider',
        'car',
        # 'truck',
        'bus',
        # 'train',
        'motorcycle',
        'bicycle',
    ])

    def __init__(self, root, transform=False, im_size=None, mean=True):
        self.root = root
        self._transform = transform
        self.im_size = im_size
        self.mean = mean
        self.ignore_label = 255
        self.n_class = len(self.rep_class_names)

        assert len(self.im_size) == 2

        dataset_dir = osp.join(self.root, 'RAND_CITYSCAPES/')
        self.files = []
        fn = os.listdir(osp.join(dataset_dir,'RGB'))

        for f in fn:
            f = f.strip()
            img_file = osp.join(dataset_dir, 'RGB/' + f)
            lbl_file = osp.join(dataset_dir, 'GT/parsed_LABELS/' + f)
            self.files.append({
                'img': img_file,
                'lbl': lbl_file,
            })

    def __getitem__(self, index):
        data_file = self.files[index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        if self.im_size:
            img = img.resize((self.im_size[0],self.im_size[1]),PIL.Image.LANCZOS)
            lbl = lbl.resize((self.im_size[0],self.im_size[1]),PIL.Image.NEAREST)
        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.int32)

        lbl_copy = self.ignore_label * np.ones(lbl.shape, dtype=np.float32)
        for k, v in self.synthia_id_to_trainid.items():
            lbl_copy[lbl == k] = v

        if self._transform:
            return self.transform(img, lbl_copy)
        else:
            return img, lbl_copy

    def __len__(self):
        return len(self.files)
        
    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        if self.mean:
            img -= self.img_mean
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        if self.mean:
            img += self.img_mean
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl

class CityScapes(data.Dataset):

    img_mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)

    class_names = np.array([
        'unlabeled','ego vehicle','rectification border','out of roi','static','dynamic',
        'ground','road','sidewalk','parking','rail track','building','wall','fence','guard rail',
        'bridge','tunnel','pole','polegroup','traffic light','traffic sign','vegetation','terrain',
        'sky','person','rider','car','truck','bus','caravan','trailer','train','motorcycle','bicycle'
    ])

    rep_class_names = np.array([
        'road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation',
        'sky','person','rider','car','bus','motorcycle','bicycle'
    ])

    cs_id_to_trainid = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
        21: 8, 23: 9, 24: 10, 25: 11, 26: 12,
        28: 13, 32: 14, 33: 15
    }

    ignore_label = 255

    def __init__(self, root, split="train", transform=False, norm=False, scale=(1, 1), val_im_size=(1024, 512), val_lbl_size=(2048, 1024), resize='none', mean=True, transformer=None):
        self.root = root
        self._transform = transform
        self.split = split
        self.val_im_size = val_im_size
        self.val_lbl_size = val_lbl_size
        self.mean = mean
        self.norm = norm
        self.n_class = len(self.rep_class_names)
        self.scale = scale
        self.transformer = transformer
        self.resize = resize
        dataset_dir = osp.join(self.root, '')
        self.files = collections.defaultdict(list)

        for sp in ["train","val","test"]:
            dirs = os.listdir(osp.join(dataset_dir,'leftImg8bit/'+sp))
            fn = []
            for d in dirs:
                fs = [d + "/" + i for i in os.listdir(osp.join(dataset_dir,'leftImg8bit/'+sp+'/'+d))]
                fn += fs
            for f in fn:
                data_file = f
                label_file = f.replace("leftImg8bit","gtFine_labelIds")
                img_file = osp.join(dataset_dir, 'leftImg8bit/' + sp + '/'+ data_file)
                lbl_file = osp.join(dataset_dir, 'gt/gtFine/' + sp + '/' + label_file)
                self.files[sp].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)

        if self.transformer:
            img, lbl = self.transformer(img, lbl)

        w, h = img.size
        if self.scale != (1., 1.):
            r = np.random.uniform(self.scale[0], self.scale[1])
            nw, nh = int(r * w), int(r * h)
            img = img.resize((nw, nh), PIL.Image.LANCZOS)
            lbl = lbl.resize((nw, nh), PIL.Image.NEAREST)
        else:
            nw, nh = w, h

        if self.resize != 'none':
            if self.resize == 'random_crop':
                img = np.array(img, dtype=np.uint8)
                lbl = np.array(lbl, dtype=np.int32)
                x_start = int(np.random.uniform(0, nw-self.im_size[0]))
                y_start = int(np.random.uniform(0, nh-self.im_size[1]))
                img = img[y_start:y_start+self.im_size[1], x_start:x_start+self.im_size[0], :]
                lbl = lbl[y_start:y_start+self.im_size[1], x_start:x_start+self.im_size[0]]
            elif self.resize == 'resize':
                img = img.resize((self.val_im_size[0], self.val_im_size[1]), PIL.Image.LANCZOS)
                lbl = lbl.resize((self.val_lbl_size[0], self.val_lbl_size[1]), PIL.Image.NEAREST)
                img = np.array(img, dtype=np.uint8)
                lbl = np.array(lbl, dtype=np.int32)
            else:
                raise NotImplementedError('Unsupported Image Resize Method: %s' % self.resize)

        lbl_copy = self.ignore_label * np.ones(lbl.shape, dtype=np.float32)
        for k, v in self.cs_id_to_trainid.items():
            lbl_copy[lbl == k] = v
        lbl = lbl_copy

        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def __len__(self):
        return len(self.files[self.split])
        
    def transform(self, img, lbl):
        img = img.astype(np.float64)
        if self.mean:
            img -= self.img_mean
        if self.norm:
            img /= 255.
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        if self.mean:
            img += self.img_mean
        if self.norm:
            img *= 255.
        img = np.clip((img + 0.5), 0, 255)
        img = img.astype(np.uint8)
        lbl = lbl.numpy()
        return img, lbl

class Synthia_with_CityScapes(data.Dataset):

    img_mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)

    class_names = np.array([
        'void','sky','Building','Road','Sidewalk','Fence',
        'Vegetation','Pole','Car','Traffic sign','Pedestrian','Bicycle','Motorcycle',
        'Parking-slot','Road-work','Traffic light','Terrain','Rider','Truck','Bus',
        'Train','Wall','Lanemarking'
    ])

    rep_class_names = np.array(['road','sidewalk','building','wall','fence','pole',
        'traffic light','traffic sign','vegetation','sky','person','rider','car',
        'bus','motorcycle','bicycle'
    ])

    few_class_index = np.array([3, 4, 5])

    # synthia_id_to_trainid = { # maybe for GTA5 - > cistyscapes
    #     3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
    #     15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
    #     8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18
    # }

    # cs_id_to_trainid = {
    #     7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    #     21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    #     28: 15, 31: 16, 32: 17, 33: 18
    # }

    synthia_id_to_trainid = {
        3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
        15: 6, 9: 7, 6: 8, 1: 9, 10: 10, 17: 11,
        8: 12, 19: 13, 12: 14, 11: 15
    }

    cs_id_to_trainid = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
        21: 8, 23: 9, 24: 10, 25: 11, 26: 12,
        28: 13, 32: 14, 33: 15
    }
    
    tgt_class_dist = [
        0.38162450,0.05470928,0.22183995,0.00742106,0.00830505,0.0150021,
        0.00199720,0.00673416,0.17530701,0.03384883,0.01316921,0.0021831,
        0.06598317,0.00394197,0.00080708,0.00718731
    ]

    src_class_dist = [
        0.1844907,0.193116,0.29294933,0.002692,0.00267898,0.01038468,0.00038941,
        0.00101269,0.10308621,0.06849977,0.04245612,0.00469237,0.04058507,
        0.01526971,0.00207064,0.0021931
    ]

    ignore_label = 255

    n_class = len(rep_class_names)

    def __init__(self, s_root, c_root, transform=False, im_size=None, mean=True, norm=False, src_transformer=None, tgt_transformer=None, tgt_transformer_2=None, resize_short=1024, k_src=1, k_tgt=1):
        self.s_root = s_root
        self.c_root = c_root
        self._transform = transform
        self.im_size = im_size
        self.mean = mean
        self.norm = norm
        self.resize_short = resize_short
        self.src_transformer = src_transformer
        self.tgt_transformer = tgt_transformer
        self.tgt_transformer_2 = tgt_transformer_2
        self.k_src = k_src
        self.k_tgt = k_tgt

        s_dataset_dir = osp.join(self.s_root, 'RAND_CITYSCAPES/')
        self.source_list = []
        self.target_list = []
        fn = os.listdir(osp.join(s_dataset_dir,'RGB'))

        for f in fn:
            f = f.strip()
            img_file = osp.join(s_dataset_dir, 'RGB/' + f)
            lbl_file = osp.join(s_dataset_dir, 'GT/parsed_LABELS/' + f)
            self.source_list.append({
                'img': img_file,
                'lbl': lbl_file,
            })


        c_dataset_dir = osp.join(self.c_root, '')

        dirs = os.listdir(osp.join(c_dataset_dir,'leftImg8bit/train'))
        fn = []
        for d in dirs:
            fs = [d + "/" + i for i in os.listdir(osp.join(c_dataset_dir,'leftImg8bit/train/'+d))]
            fn += fs
        for f in fn:
            data_file = f
            label_file = f.replace("leftImg8bit","gtFine_labelIds")
            img_file = osp.join(c_dataset_dir, 'leftImg8bit/train/'+ data_file)
            lbl_file = osp.join(c_dataset_dir, 'gt/gtFine/train/' + label_file)
            self.target_list.append({
                'img': img_file,
                'lbl': lbl_file,
            })
        self.source_list = np.array(self.source_list)
        self.target_list = np.array(self.target_list)

        l1 = self.source_list.shape[0]
        l2 = self.target_list.shape[0]
        _sources = [self.source_list[np.random.permutation(l1)] for i in range(self.k_src)]
        _targets = [self.target_list[np.random.permutation(l2)] for i in range(self.k_tgt)]
        p1 = np.random.permutation(l1)
        p2 = np.random.permutation(l2)
        p1_ = np.random.permutation(l1)
        _source = self.source_list[p1]
        _target = self.target_list[p2]
        _source_2 = self.source_list[p1_]
        if l1 == l2:
            self.src_files = np.stack(_sources,1)
            self.tgt_files = np.stack(_targets,1)
        if l2 > l1:
            times = int(l2 / l1) + 1
            times_sources = [s.repeat(times, axis=0)[:l2] for s in _sources]
            self.src_files = np.stack(times_sources,1)
            self.tgt_files = np.stack(_targets,1)
        if l1 > l2:
            times = int(l1 / l2) + 1
            times_targets = [t.repeat(times, axis=0)[:l1] for t in _targets]
            self.src_files = np.stack(_sources,1)
            self.tgt_files = np.stack(times_targets,1)

    def __getitem__(self, index):
        src_data_file = self.src_files[index]
        tgt_data_file = self.tgt_files[index]

        source_img_file = src_data_file[0]['img']
        source_img = PIL.Image.open(source_img_file)

        source_lbl_file = src_data_file[0]['lbl']
        source_lbl = PIL.Image.open(source_lbl_file)

        target_img_file = tgt_data_file[0]['img']
        target_img = PIL.Image.open(target_img_file)

        target_lbl_file = tgt_data_file[0]['lbl']
        target_lbl = PIL.Image.open(target_lbl_file)

        source_full_imgs_file = [i['img'] for i in src_data_file]
        source_full_imgs = [PIL.Image.open(i) for i in source_full_imgs_file]

        target_full_imgs_file = [i['img'] for i in tgt_data_file]
        target_full_imgs = [PIL.Image.open(i) for i in target_full_imgs_file]

        if self.src_transformer:
            source_img, source_lbl = self.src_transformer(source_img, source_lbl)

        if self.tgt_transformer:
            target_img_1, target_img_2, target_lbl_1, target_lbl_2, transform_param1, transform_param2 = self.tgt_transformer(target_img, target_img, target_lbl, target_lbl)
        else:
            target_img_1, target_img_2 ,target_lbl_1, target_lbl_2 = target_img, target_img, target_lbl, target_lbl
            transform_param1, transform_param2 = np.array([]), np.array([])

        sw, sh = source_img.size
        tw, th = target_img_1.size
        
        src_ratio = self.resize_short / min(sw, sh)
        tgt_ratio = self.resize_short / min(tw, th)
        source_img = source_img.resize((int(sw * src_ratio), int(sh * src_ratio)))
        source_lbl = source_lbl.resize((int(sw * src_ratio), int(sh * src_ratio)))
        target_img_1 = target_img_1.resize((int(tw * tgt_ratio), int(th * tgt_ratio)))
        target_lbl_1 = target_lbl_1.resize((int(tw * tgt_ratio), int(th * tgt_ratio)))
        target_img_2 = target_img_2.resize((int(tw * tgt_ratio), int(th * tgt_ratio)))
        target_lbl_2 = target_lbl_2.resize((int(tw * tgt_ratio), int(th * tgt_ratio)))

        sw, sh = source_img.size
        tw, th = target_lbl_1.size

        source_img = np.array(source_img, dtype=np.uint8)
        target_img_1 = np.array(target_img_1, dtype=np.uint8)
        target_img_2 = np.array(target_img_2, dtype=np.uint8)
        source_lbl = np.array(source_lbl, dtype=np.int32)
        target_lbl_1 = np.array(target_lbl_1, dtype=np.int32)
        target_lbl_2 = np.array(target_lbl_2, dtype=np.int32)
        if self.im_size:

            x_start = int(np.random.uniform(0, sw-self.im_size[0]))
            y_start = int(np.random.uniform(0, sh-self.im_size[1]))
            source_img = source_img[y_start:y_start+self.im_size[1], x_start:x_start+self.im_size[0], :]
            source_lbl = source_lbl[y_start:y_start+self.im_size[1], x_start:x_start+self.im_size[0]]

            x_start = int(np.random.uniform(0, tw-self.im_size[0]))
            y_start = int(np.random.uniform(0, th-self.im_size[1]))
            target_img_1 = target_img_1[y_start:y_start+self.im_size[1], x_start:x_start+self.im_size[0], :]
            target_img_2 = target_img_2[y_start:y_start+self.im_size[1], x_start:x_start+self.im_size[0], :]
            target_lbl_1 = target_lbl_1[y_start:y_start+self.im_size[1], x_start:x_start+self.im_size[0]]
            target_lbl_2 = target_lbl_2[y_start:y_start+self.im_size[1], x_start:x_start+self.im_size[0]]

            source_full_imgs = [s.resize((self.im_size[0], self.im_size[1])) for s in source_full_imgs]
            target_full_imgs = [t.resize((self.im_size[0], self.im_size[1])) for t in target_full_imgs]

        source_full_imgs = [np.array(s, dtype=np.uint8) for s in source_full_imgs] # should nest
        target_full_imgs = [np.array(t, dtype=np.uint8) for t in target_full_imgs]

        source_lbl_copy = self.ignore_label * np.ones(source_lbl.shape, dtype=np.float32)
        
        for k, v in self.synthia_id_to_trainid.items():
            source_lbl_copy[source_lbl == k] = v

        tgt_lbl_1_copy = self.ignore_label * np.ones(target_lbl_1.shape, dtype=np.float32)
        for k, v in self.cs_id_to_trainid.items():
            tgt_lbl_1_copy[target_lbl_1 == k] = v

        tgt_lbl_2_copy = self.ignore_label * np.ones(target_lbl_2.shape, dtype=np.float32)
        for k, v in self.cs_id_to_trainid.items():
            tgt_lbl_2_copy[target_lbl_2 == k] = v

        if self._transform:
            source_img = self.tranform_img(source_img)
            target_img_1 = self.tranform_img(target_img_1)
            target_img_2 = self.tranform_img(target_img_2)
            source_lbl_copy = self.transform_lbl(source_lbl_copy)
            tgt_lbl_1_copy = self.transform_lbl(tgt_lbl_1_copy)
            tgt_lbl_2_copy = self.transform_lbl(tgt_lbl_2_copy)
            source_full_imgs = torch.stack([self.tranform_img(s) for s in source_full_imgs])
            target_full_imgs = torch.stack([self.tranform_img(t) for t in target_full_imgs])

            return source_img, target_img_1, target_img_2, source_lbl_copy, tgt_lbl_1_copy, tgt_lbl_2_copy, torch.from_numpy(transform_param1), torch.from_numpy(transform_param2), source_full_imgs, target_full_imgs
        else:
            return source_img, target_img_1, target_img_2, source_lbl_copy, tgt_lbl_1_copy, tgt_lbl_2_copy, torch.from_numpy(transform_param1), torch.from_numpy(transform_param2), source_full_imgs, target_full_imgs

    def __len__(self):
        return len(self.src_files)
        
    def tranform_img(self, img):
        img = img.astype(np.float64)
        if self.mean:
            img -= self.img_mean
        if self.norm:
            img = img / (255.)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img

    def transform_lbl(self, lbl):
        lbl = torch.from_numpy(lbl).long()
        return lbl

    def untransform(self, img):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        if self.norm:
            img *= 255.
        if self.mean:
            img += self.img_mean
        img = np.clip((img + 0.5), 0, 255)
        img = img.astype(np.uint8)
        return img

    def fit_image(self, img, cuda=True):

        if self.norm:
            img = img * 255.
        if self.mean:
            mean_tensor = torch.from_numpy(self.img_mean).view(1,3,1,1)
            if cuda:
                mean_tensor = mean_tensor.cuda()
            img = img + mean_tensor

        img = img.add(0.5).clamp(0, 255)

        if self.mean:
            mean_tensor = torch.from_numpy(self.img_mean).view(1,3,1,1)
            if cuda:
                mean_tensor = mean_tensor.cuda()
            img = img - mean_tensor
        if self.norm:
            img = img / 255.

        return img

