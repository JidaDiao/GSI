import argparse
import os
import timm
import torch as t
import numpy as np
import torch.nn as nn
import cv2
import torchattacks_local as ta
from utils import yaml_config_hook
from GSI_layer import Add_norm, Add_GSInorm
from Tools import shows, trans

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZK")
    config = yaml_config_hook('./config/config.yaml')
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    imgs = args.image
    img = cv2.imread(imgs)
    x = trans(img)
    x = t.unsqueeze(x, dim=0).to(device)
    model = timm.create_model(args.model, pretrained=True).to(device)
    modelgsi = Add_GSInorm(model).to(device)
    modelnorm = Add_norm(model).to(device)
    for p in modelnorm.parameters():
        p.requires_grad = False
    for p in modelgsi.parameters():
        p.requires_grad = False
    atname = args.attack
    dirname = args.name
    root = './result'
    folder = os.path.join(root, atname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 普通攻击
    attackdic = {'FGSM': ta.FGSM(modelnorm, eps=eval(args.eps)),
                 'I-FGSM': ta.BIM(modelnorm, steps=args.steps, eps=eval(args.eps)),
                 'CW': ta.CW(modelnorm, c=eval(args.c))}
    modelnorm.eval()
    shows(x, 'origin_{}.png'.format(atname), folder)
    y = modelnorm(x)
    soft = nn.Softmax(dim=1)
    y = soft(y)
    print(y[:, int(np.argmax(y.data.cpu().numpy(), axis=1))])
    print(np.argmax(y.data.cpu().numpy(), axis=1))
    attack = attackdic[atname]
    lab = t.tensor([int(args.label)], dtype=t.long).to(device)
    x_adversarial = attack(x, lab.long()).to(device)
    noise = x_adversarial - x
    x_adversarial = t.clamp(x_adversarial, min=0, max=1)
    shows(x_adversarial, 'adv1_{}.png'.format(dirname), folder)
    shows(noise, 'noise1_{}.png'.format(dirname), folder, flag=True)
    y = modelnorm(x_adversarial)
    soft = nn.Softmax(dim=1)
    y = soft(y)
    print(y[:, int(np.argmax(y.data.cpu().numpy(), axis=1))])
    print(np.argmax(y.data.cpu().numpy(), axis=1))

    # 梯度反转攻击
    attackgsidic = {'FGSM': ta.FGSM(modelgsi, eps=eval(args.eps)),
                    'I-FGSM': ta.BIM(modelgsi, steps=args.steps, eps=eval(args.eps)),
                    'CW': ta.CW(modelgsi, c=eval(args.c))}
    modelgsi.eval()
    attack2 = attackgsidic[atname]
    x_adversarial2 = attack2(x, lab.long()).to(device)
    noise2 = x_adversarial2 - x
    x_adversarial2 = t.clamp(x_adversarial2, min=0, max=1)
    shows(x_adversarial2, 'adv2_{}.png'.format(dirname), folder)
    shows(noise2, 'noise2_{}.png'.format(dirname), folder, flag=True)
    y = modelgsi(x_adversarial2)
    soft = nn.Softmax(dim=1)
    y = soft(y)
    print(y[:, int(np.argmax(y.data.cpu().numpy(), axis=1))])
    print(np.argmax(y.data.cpu().numpy(), axis=1))

    # show
    noisesum = noise + noise2
    shows(noisesum * 255, 'noisesum{}.png'.format(dirname), folder, flag=True)
