from torchvision.utils import save_image
import os
import torchvision.transforms as transforms


def shows(img, name, root, flag=False):
    if flag:
        saveroot = os.path.join(root, name)
        img = img[:, (2, 1, 0), :, :]
        save_image(img * 255, saveroot)
    else:
        saveroot = os.path.join(root, name)
        img = img[:, (2, 1, 0), :, :]
        save_image(img, saveroot)


trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([224, 224]),
    transforms.ToTensor()
])
