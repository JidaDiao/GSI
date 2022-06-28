import argparse
import math
import numpy as np
import torch as t
import timm
import torchattacks as ta
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
from utils import yaml_config_hook

train_transform224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
train_transform299 = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])
batch_size = 32
num_worker_size = 8
img_root = '/data/ImageNet2012/val/'

val_dataset224 = torchvision.datasets.ImageFolder(
    root=img_root,
    transform=train_transform224)
val_loader224 = DataLoader(val_dataset224, batch_size=batch_size, shuffle=True, num_workers=num_worker_size,
                           pin_memory=True)
val_dataset299 = torchvision.datasets.ImageFolder(
    root=img_root,
    transform=train_transform299)
val_loader299 = DataLoader(val_dataset299, batch_size=batch_size, shuffle=True, num_workers=num_worker_size,
                           pin_memory=True)


class Add_norm(nn.Module):
    def __init__(self, Omodel):
        super(Add_norm, self).__init__()
        self.Omodel = Omodel
        self.mean = nn.Parameter(t.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = nn.Parameter(t.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.Omodel(x)
        return x


class Add_GSInorm(nn.Module):
    def __init__(self, Omodel):
        super(Add_GSInorm, self).__init__()
        self.Omodel = Omodel
        self.mean = nn.Parameter(t.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = nn.Parameter(t.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        with t.no_grad():
            x_t = t.clamp(t.round(x * 255), 0, 255) / 255
            T = x_t - x
        x = T + x
        x = t.sin(x * 510 * math.pi + math.pi) + x
        x = (x - self.mean) / self.std
        x = self.Omodel(x)
        return x


transtorand = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ZK")
    config = yaml_config_hook('./config/config.yaml')
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    modellist = args.modellist
    modellist299 = args.modellist299
    attacklist = args.mainattack
    print("Size of Test data = {}".format(val_dataset224.__len__()))
    for n, name in enumerate(modellist):
        model = timm.create_model(name, pretrained=True).cuda()
        model = Add_GSInorm(model).cuda()
        model.eval()
        attackset = {'FGSM-2/255': ta.FGSM(model, eps=2 / 255),
                     'FGSM-4/255': ta.FGSM(model, eps=4 / 255),
                     'I-FGSM-2/255-5': ta.BIM(model, steps=5, eps=2 / 255),
                     'I-FGSM-2/255-10': ta.BIM(model, steps=10, eps=2 / 255),
                     'I-FGSM-4/255-5': ta.BIM(model, steps=5, eps=4 / 255),
                     'I-FGSM-4/255-10': ta.BIM(model, steps=10, eps=4 / 255),
                     'I-FGSM-8/255-5': ta.BIM(model, steps=5, eps=8 / 255),
                     'I-FGSM-8/255-10': ta.BIM(model, steps=10, eps=8 / 255),
                     'CW': ta.CW(model)}
        for i, v in enumerate(attacklist):
            attack = attackset[v]
            ar_acc = 0
            clean_acc = 0
            adv_acc = 0
            am_err = 0
            clean_err = 0
            for i, data in tqdm(enumerate(val_loader224)):
                img = data[0]
                clean_pred = model(img.cuda(non_blocking=True))
                clean_predt = np.argmax(clean_pred.cpu().data.numpy(), axis=1) == data[1].numpy()
                clean_acc += np.sum(clean_predt)
                err_predt = np.argmax(clean_pred.cpu().data.numpy(), axis=1) != data[1].numpy()
                clean_err += np.sum(err_predt)
                img = attack(data[0], data[1].long())
                adv_pred = model(img.cuda(non_blocking=True))
                adv_predt = np.argmax(adv_pred.cpu().data.numpy(), axis=1) == data[1].numpy()
                adv_acc += np.sum(adv_predt)
                ar_acc += np.sum(adv_predt * clean_predt)
                am_err += np.sum(err_predt * adv_predt)
            print('{}: {}_norm: AR = {:.2f} , ACC = {:.2f}, AM = {:.2f}'.format(v, name, 100 * ar_acc / clean_acc,
                                                                                100 * adv_acc / val_dataset224.__len__(),
                                                                                100 * am_err / clean_err))

    for n, name in enumerate(modellist299):
        model = timm.create_model(name, pretrained=True).cuda()
        model = Add_GSInorm(model).cuda()
        model.eval()
        attackset = {'FGSM-2/255': ta.FGSM(model, eps=2 / 255),
                     'FGSM-4/255': ta.FGSM(model, eps=4 / 255),
                     'I-FGSM-2/255-5': ta.BIM(model, steps=5, eps=2 / 255),
                     'I-FGSM-2/255-10': ta.BIM(model, steps=10, eps=2 / 255),
                     'I-FGSM-4/255-5': ta.BIM(model, steps=5, eps=4 / 255),
                     'I-FGSM-4/255-10': ta.BIM(model, steps=10, eps=4 / 255),
                     'I-FGSM-8/255-5': ta.BIM(model, steps=5, eps=8 / 255),
                     'I-FGSM-8/255-10': ta.BIM(model, steps=10, eps=8 / 255),
                     'CW': ta.CW(model)}
        for i, v in enumerate(attacklist):
            attack = attackset[v]
            ar_acc = 0
            clean_acc = 0
            adv_acc = 0
            am_err = 0
            clean_err = 0
            for i, data in tqdm(enumerate(val_loader299)):
                img = data[0]
                clean_pred = model(img.cuda(non_blocking=True))
                clean_predt = np.argmax(clean_pred.cpu().data.numpy(), axis=1) == data[1].numpy()
                clean_acc += np.sum(clean_predt)
                err_predt = np.argmax(clean_pred.cpu().data.numpy(), axis=1) != data[1].numpy()
                clean_err += np.sum(err_predt)
                img = attack(data[0], data[1].long())
                adv_pred = model(img.cuda(non_blocking=True))
                adv_predt = np.argmax(adv_pred.cpu().data.numpy(), axis=1) == data[1].numpy()
                adv_acc += np.sum(adv_predt)
                ar_acc += np.sum(adv_predt * clean_predt)
                am_err += np.sum(err_predt * adv_predt)
            print('{}: {}_norm: AR = {:.2f} , ACC = {:.2f}, AM = {:.2f}'.format(v, name, 100 * ar_acc / clean_acc,
                                                                                100 * adv_acc / val_dataset224.__len__(),
                                                                                100 * am_err / clean_err))

    print('--------------------NOT-GSI------------------------')

    for n, name in enumerate(modellist):
        model = timm.create_model(name, pretrained=True).cuda()
        model = Add_norm(model).cuda()
        model.eval()
        attackset = {'FGSM-2/255': ta.FGSM(model, eps=2 / 255),
                     'FGSM-4/255': ta.FGSM(model, eps=4 / 255),
                     'I-FGSM-2/255-5': ta.BIM(model, steps=5, eps=2 / 255),
                     'I-FGSM-2/255-10': ta.BIM(model, steps=10, eps=2 / 255),
                     'I-FGSM-4/255-5': ta.BIM(model, steps=5, eps=4 / 255),
                     'I-FGSM-4/255-10': ta.BIM(model, steps=10, eps=4 / 255),
                     'I-FGSM-8/255-5': ta.BIM(model, steps=5, eps=8 / 255),
                     'I-FGSM-8/255-10': ta.BIM(model, steps=10, eps=8 / 255),
                     'CW': ta.CW(model)}
        for i, v in enumerate(attacklist):
            attack = attackset[v]
            ar_acc = 0
            clean_acc = 0
            adv_acc = 0
            am_err = 0
            clean_err = 0
            for i, data in tqdm(enumerate(val_loader224)):
                img = data[0]
                clean_pred = model(img.cuda(non_blocking=True))
                clean_predt = np.argmax(clean_pred.cpu().data.numpy(), axis=1) == data[1].numpy()
                clean_acc += np.sum(clean_predt)
                err_predt = np.argmax(clean_pred.cpu().data.numpy(), axis=1) != data[1].numpy()
                clean_err += np.sum(err_predt)
                img = attack(data[0], data[1].long())
                adv_pred = model(img.cuda(non_blocking=True))
                adv_predt = np.argmax(adv_pred.cpu().data.numpy(), axis=1) == data[1].numpy()
                adv_acc += np.sum(adv_predt)
                ar_acc += np.sum(adv_predt * clean_predt)
                am_err += np.sum(err_predt * adv_predt)
            print('{}: {}_norm: AR = {:.2f} , ACC = {:.2f}, AM = {:.2f}'.format(v, name, 100 * ar_acc / clean_acc,
                                                                                100 * adv_acc / val_dataset224.__len__(),
                                                                                100 * am_err / clean_err))
    for n, name in enumerate(modellist299):
        model = timm.create_model(name, pretrained=True).cuda()
        model = Add_norm(model).cuda()
        model.eval()
        attackset = {'FGSM-2/255': ta.FGSM(model, eps=2 / 255),
                     'FGSM-4/255': ta.FGSM(model, eps=4 / 255),
                     'I-FGSM-2/255-5': ta.BIM(model, steps=5, eps=2 / 255),
                     'I-FGSM-2/255-10': ta.BIM(model, steps=10, eps=2 / 255),
                     'I-FGSM-4/255-5': ta.BIM(model, steps=5, eps=4 / 255),
                     'I-FGSM-4/255-10': ta.BIM(model, steps=10, eps=4 / 255),
                     'I-FGSM-8/255-5': ta.BIM(model, steps=5, eps=8 / 255),
                     'I-FGSM-8/255-10': ta.BIM(model, steps=10, eps=8 / 255),
                     'CW': ta.CW(model)}
        for i, v in enumerate(attacklist):
            attack = attackset[v]
            ar_acc = 0
            clean_acc = 0
            adv_acc = 0
            am_err = 0
            clean_err = 0
            for i, data in tqdm(enumerate(val_loader299)):
                img = data[0]
                clean_pred = model(img.cuda(non_blocking=True))
                clean_predt = np.argmax(clean_pred.cpu().data.numpy(), axis=1) == data[1].numpy()
                clean_acc += np.sum(clean_predt)
                err_predt = np.argmax(clean_pred.cpu().data.numpy(), axis=1) != data[1].numpy()
                clean_err += np.sum(err_predt)
                img = attack(data[0], data[1].long())
                adv_pred = model(img.cuda(non_blocking=True))
                adv_predt = np.argmax(adv_pred.cpu().data.numpy(), axis=1) == data[1].numpy()
                adv_acc += np.sum(adv_predt)
                ar_acc += np.sum(adv_predt * clean_predt)
                am_err += np.sum(err_predt * adv_predt)
            print('{}: {}_norm: AR = {:.2f} , ACC = {:.2f}, AM = {:.2f}'.format(v, name, 100 * ar_acc / clean_acc,
                                                                                100 * adv_acc / val_dataset224.__len__(),
                                                                                100 * am_err / clean_err))
