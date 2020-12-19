import argparse
import sys
sys.path.append('/path/to/project/adain')
from pathlib import Path
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm
import datasets
import net
from torchvision.utils import save_image
import se_transformers
import cv2
import skimage.io
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_model_dir', default='./saved_model',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=40000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=0.1)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--log_img_interval', type=int, default=100)
args = parser.parse_args()

s_root = '/path/to/data/synthia'
c_root = '/path/to/data/cityscape'

train_im_size = (640,320)
resize_short=760
src_transformer = se_transformers.Src_Compose([
    # se_transformers.Src_Horizontal_Flip(),
    # se_transformers.Src_RandomAffine(degrees=0, translate=(0.02, 0.02), shear=20),
    se_transformers.Src_jitter(brightness=0, contrast=0, saturation=0),
])
tgt_transformer = se_transformers.Tgt_Compose([
    # se_transformers.Tgt_Horizontal_Flip(),
    # se_transformers.Tgt_RandomAffine(degrees=5, translate=(0.02, 0.02), shear=20),
    se_transformers.Tgt_jitter(brightness=0, contrast=0, saturation=0, brightness_2=0, contrast_2=0, saturation_2=0),
])
transformer = transforms.Compose([
    transforms.ToTensor()
])

train_loader = torch.utils.data.DataLoader(
        datasets.Synthia_with_CityScapes(s_root, c_root, transform=True, im_size=train_im_size, src_transformer=src_transformer, tgt_transformer=tgt_transformer, resize_short=resize_short, k_src=1),
        batch_size=args.batch_size, shuffle=True)

device = torch.device('cuda')
log_root = "adain/logs_adain/"

save_model_dir = Path(os.path.join(log_root, args.save_model_dir))
save_model_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(os.path.join(log_root, args.log_dir))
log_dir.mkdir(exist_ok=True, parents=True)

exp_name = "0_1"
fname = os.path.join(log_dir, "log_" + exp_name + ".txt")
out = os.path.join(log_root, "save_imgs/save_img_" + exp_name + "/")
if not os.path.exists(out):
    os.makedirs(out)
f = open(fname, "w")
f.close()

decoder = net.decoder
# decoder_pretrained_path = 'experiments/decoder__style_transfer_2_new_dataset_0_01_640_320_d5_mean_nonorm.pth.tar'
# decoder.load_state_dict(torch.load(decoder_pretrained_path))
vgg = net.vgg
vgg_pretrained_path = 'saved_models/vgg_normalised.pth'
vgg.load_state_dict(torch.load(vgg_pretrained_path))
vgg = nn.Sequential(*list(vgg.children())[:31])
transfer_network = net.Net(vgg, decoder)
transfer_network.train()
transfer_network.to(device)
decoder_optimizer = torch.optim.Adam(transfer_network.decoder.parameters(), lr=args.lr)


i = 0

max_epoch = int(args.max_iter / len(train_loader))
for e in range(max_epoch):
    for source_image, target_image, _, _, _, _, _, _, source_image_full, target_image_full in train_loader:
        if np.random.rand() > 0.5:
            content_images = source_image
            style_images = target_image_full.squeeze()
        else:
            content_images = target_image
            style_images = source_image_full.squeeze()
        content_images = content_images.to(device).float() 
        style_images = style_images.to(device).float()

        # decoder
        loss_c, loss_s, g_t = transfer_network(content_images, style_images)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        decoder_loss = loss_c + loss_s
        decoder_optimizer.zero_grad()
        decoder_loss.backward()
        decoder_optimizer.step()

        # log
        f = open(fname,"a")
        report = "iter: " + str(i) + ", decoder_loss: " + str(decoder_loss.item()) +  ", content loss: " + str(loss_c.item()) +  ", style loss: " + str(loss_s.item())
        f.write(report + '\n')
        print(report)
        if i % args.log_img_interval == 0:
            skimage.io.imsave(out + str(i) + ".png", np.concatenate((
                (train_loader.dataset.untransform(g_t[0].detach().cpu())), 
                (train_loader.dataset.untransform(content_images[0].detach().cpu())), 
                (train_loader.dataset.untransform(style_images[0].detach().cpu()))), axis=1))

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = net.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            save_name = os.path.join(save_model_dir, "decoder_" + exp_name + ".pth.tar")
            torch.save(state_dict, save_name)
        i += 1
        if i >= args.max_iter:
            break

