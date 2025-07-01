import importlib
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from utils.option import args
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)


def main_worker(args, use_gpu=True):
    # device = torch.device("cuda") if use_gpu else torch.device("cpu")

    # Model and version
    net = importlib.import_module("model." + args.model)
    model = net.InpaintGenerator(args).cuda()
    model.load_state_dict(torch.load(args.pre_train, map_location="cuda",weights_only=True))
    model.eval()

    # prepare dataset
    image_paths = []
    for ext in [".jpg", ".png"]:
        image_paths.extend(glob(os.path.join(args.dir_image, "*" + ext)))
    image_paths.sort()
    mask_paths = sorted(glob(os.path.join(args.dir_mask, "*.png")))
    os.makedirs(args.outputs, exist_ok=True)

    # iteration through datasets
    for ipath, mpath in zip(image_paths, mask_paths):
        image = (Image.open(ipath).convert("RGB"))
        
        img000 = np.array(image)
        image = ToTensor()(image)
        image = (image * 2.0 - 1.0).unsqueeze(0)

        mask = (Image.open(mpath).convert("L"))
        #print("eee",image.size,mask.size)

        mask000 = np.array(mask)
        mask = ToTensor()(mask) 
        mask = mask.unsqueeze(0)
        mask = F.interpolate(mask, size=image.shape[2:], mode='bilinear', align_corners=False)
        
        plt.figure(figsize=(12,4),dpi=80)
        plt.subplot(1, 3, 1)
        plt.imshow(img000)
        plt.subplot(1, 3, 2)
        plt.imshow(mask000)
        plt.imshow(mask.squeeze(), cmap='gray')
        #plt.show()

        image, mask = image.cuda(), mask.cuda()
        
        image_masked = image * (1 - mask.float()) + mask  #before
        #image_masked = image *mask.float()
        #print(image_masked.shape)
        with torch.no_grad():
            pred_img = model(image_masked, mask)
        #print(pred_img.shape)
        #print(image.shape)
        #print(mask.shape)
        target_size = image.size()[2:]
        pred_img = F.interpolate( pred_img, size=target_size, mode='bilinear', align_corners=False)

        comp_imgs = (1 - mask) * image + mask * pred_img #before
        #comp_imgs = mask.float() * image + (1- mask.float()) * pred_img
        image_name = os.path.basename(ipath).split(".")[0]
        postprocess(image_masked[0]).save(os.path.join(args.outputs, f"{image_name}_masked.png"))
        postprocess(pred_img[0]).save(os.path.join(args.outputs, f"{image_name}_pred.png"))
        postprocess(comp_imgs[0]).save(os.path.join(args.outputs, f"{image_name}_comp.png"))
        print(f"saving to {os.path.join(args.outputs, image_name)}")


if __name__ == "__main__":
    #args.dir_image=r"E:\liufuzhan\Documents\AOT-gan\test\test\images"
    args.dir_image=r"E:\liufuzhan\Documents\AOT-gan\AOT-GAN-for-Inpainting\src\test_data\image"
    #args.dir_mask=r"E:\liufuzhan\Documents\AOT-gan\test\test\masks" #others
    args.dir_mask=r"E:\liufuzhan\Documents\AOT-gan\AOT-GAN-for-Inpainting\src\test_data\mask"
    args.pre_train=r"C:\Users\liufuzhan\Desktop\G0021000\G0021000.pt"
    args.outputs = "../outputs/our21000_hou500zhang"
    args.painter="bbox" #'freeform', 'bbox'
    main_worker(args)
