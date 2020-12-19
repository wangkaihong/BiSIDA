import sys
sys.path.append('/path/to/project/')
import torch
from dataloader import datasets
from models import fcn
from models import Style_net
import os
from trainer import trainer
import torch.nn as nn
from optimization import optim_weight_ema
from dataloader import se_transformers
from torchvision import transforms
import numpy as np

torch.manual_seed(2)
np.random.seed(2)
torch.cuda.manual_seed(2)
torch.backends.cudnn.deterministic=True

s_root = '/path/to/data/gta5'
c_root = '/path/to/data/cityscape'

def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        fcn.FCN32s,
        fcn.FCN16s,
        fcn.FCN8s,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))
start_epoch = 0
start_iteration = 0

train_im_size = (960, 480)
resize_short=760
val_im_size = (1024, 512)
val_lbl_size=(2048, 1024)

src_transformer = se_transformers.Src_Compose([
    # se_transformers.Src_Horizontal_Flip(),
    # se_transformers.Src_RandomAffine(degrees=0, translate=(0.02, 0.02), shear=20),
    se_transformers.Src_jitter(brightness=0.2, contrast=0.2),
])
tgt_transformer = se_transformers.Tgt_Compose([
    # se_transformers.Tgt_Horizontal_Flip(),
    # se_transformers.Tgt_RandomAffine(degrees=5, translate=(0.02, 0.02), shear=20),
    se_transformers.Tgt_jitter(brightness=0.2, contrast=0.2),
])

gpus = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
k_src = 4
k_tgt = 1

train_loader = torch.utils.data.DataLoader(
        datasets.GTAV_with_CityScapes(s_root, c_root, transform=True, im_size=train_im_size, src_transformer=src_transformer, tgt_transformer=tgt_transformer, resize_short=resize_short, k_src=k_src, k_tgt=k_tgt),
        batch_size=len(gpus), shuffle=True, num_workers=1, drop_last=True)

val_loader = torch.utils.data.DataLoader(
        datasets.CityScapes(c_root, split="val", transform=True, val_im_size=val_im_size, val_lbl_size=val_lbl_size, resize='resize', transformer=None, validate_type='gtav'),
        batch_size=1, shuffle=True)

n_class = train_loader.dataset.n_class

lr = 1.0e-5
weight_decay = 0.0005
momentum = 0.99
teacher_alpha = 0.999
loss_type='ce'
size_average=True 
class_dist_reg=True
interval_validate=3000
unsup_weight = 1
style_alpha = 0.5
# def rampup_function(weight, step):
#     return weight * (np.e ** (-5 * ((1 - step) ** 2)))
rampup_function=None
confidence_thresh=0.9
balance_function=lambda x: pow((1 / x),0.17) / 1.
internal_weight=None
clustering_weight = 0
pad = 50
pseudo_labeling = True
src_style_alpha = [0, 1]
tgt_style_alpha = [0, 1]
src_transfer_rate = 0.5
tgt_transfer_rate = 0.5
tgt_style_method = "only_2"
source_record = False
train_generator = False
style_weight = 1e-5
content_weight = 1e-5
src_ave_method="ave_"
tgt_ave_method="ave_"
src_temperture=1
tgt_temperture=1./4.
betas=(0.99, 0.999)

out = "logs/train_gta_vgg_experiment"
if not os.path.exists(out):
    os.makedirs(out)
vgg16 = fcn.VGG16(pretrained=True)

student = fcn.FCN8sAtOnce(n_class=n_class)
student.copy_params_from_vgg16(vgg16)

teacher = fcn.FCN8sAtOnce(n_class=n_class)
teacher.copy_params_from_vgg16(vgg16)

decoder = Style_net.decoder
decoder_pretrained_path = 'saved_models/decoder_gta_0_1.pth.tar'
decoder.load_state_dict(torch.load(decoder_pretrained_path))
vgg = Style_net.vgg
vgg_pretrained_path = 'saved_models/vgg_normalised.pth'
vgg.load_state_dict(torch.load(vgg_pretrained_path))
vgg = nn.Sequential(*list(vgg.children())[:31])
style_net = Style_net.Net(vgg, decoder)
style_net_optim = torch.optim.Adam(style_net.decoder.parameters(), lr=lr)

# fcn16s = fcn.FCN16s()
# state_dict = torch.load(fcn.FCN16s.download())
# try:
#     fcn16s.load_state_dict(state_dict)
# except RuntimeError:
#     fcn16s.load_state_dict(state_dict['model_state_dict'])
# model.copy_params_from_fcn16s(fcn16s)

stu_optim = torch.optim.Adam(
    [
        {'params': get_parameters(student, bias=False)},
        {'params': get_parameters(student, bias=True),
         'lr': lr * 2, 'weight_decay': 0},
    ],
    lr=lr,
    betas=betas,
    weight_decay=weight_decay)

tea_optim = optim_weight_ema.OldWeightEMA(teacher, student, alpha=teacher_alpha)

cuda = torch.cuda.is_available()

if cuda:
    student = student.cuda()
    teacher = teacher.cuda()
    style_net = style_net.cuda()
max_iteration = 400000



trainer = trainer.Trainer(
    cuda=cuda,
    student_model=student,
    teacher_model=teacher,
    style_net=style_net,
    student_optimizer=stu_optim,
    teacher_optimizer=tea_optim,
    train_loader=train_loader,
    val_loader=val_loader,
    out=out,
    source_record=source_record,
    src_ave_method=src_ave_method,
    tgt_ave_method=tgt_ave_method,
    src_temperture=src_temperture,
    tgt_temperture=tgt_temperture,    
    train_generator=train_generator,
    style_weight=style_weight,
    content_weight=content_weight,
    style_net_optim=style_net_optim,
    max_iter=max_iteration,
    unsup_weight=unsup_weight,
    clustering_weight=clustering_weight,
    loss_type=loss_type, 
    pseudo_labeling=pseudo_labeling,
    balance_function=balance_function,
    confidence_thresh=confidence_thresh,
    internal_weight=internal_weight,
    size_average=size_average, 
    class_dist_reg=class_dist_reg, 
    interval_validate=interval_validate,
    rampup_function=rampup_function,
    src_style_alpha = src_style_alpha,
    tgt_style_alpha = tgt_style_alpha,
    src_transfer_rate = src_transfer_rate,
    tgt_transfer_rate = tgt_transfer_rate,
    tgt_style_method = tgt_style_method,
    pad=pad
)
trainer.epoch = start_epoch
trainer.iteration = start_iteration
trainer.train()

