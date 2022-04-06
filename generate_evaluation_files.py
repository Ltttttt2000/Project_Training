"""
plot the ssim, psnr
save value in csv files.

"""
import os

import numpy as np
import torch.nn
from PIL import Image
from skimage.metrics import structural_similarity as sk_cpt_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import csv


def imageList(path):
    image_list = dict()
    # for i in range(0, 100):
    #     image_list.setdefault(i, [])
    for img in os.listdir(path):
        index_epoch = img.split('-')
        index = index_epoch[0]
        epoch = index_epoch[1].replace('.png', '')
        # print(index, epoch)
        image_list.setdefault(epoch, []).append(index)
    return image_list


def calculate_ssim(images, path1, path2, file_name):
    with open(file_name, "w") as ssim_file:
        ssim_writer = csv.writer(ssim_file)
        for key, values in images.items():
            # print(key, values)
            name = values[0] + '-' + key + '.png'
            img1 = np.array(Image.open(path1 + name).convert('RGB'))
            img2 = np.array(Image.open(path2 + name).convert('RGB'))
            name = values[1] + '-' + key + '.png'
            img3 = np.array(Image.open(path1 + name).convert('RGB'))
            img4 = np.array(Image.open(path2 + name).convert('RGB'))
            # print(path1)
            ssim1 = sk_cpt_ssim(img1, img2, multichannel=True)
            ssim2 = sk_cpt_ssim(img3, img4, multichannel=True)

            ssim = (ssim1 + ssim2) / 2
            ssim_writer.writerow([key, ssim])


'''
images: the dict store [key-values]: [epoch-index]
path1: the first path to images
path2: the second path to images
file_name: the csv file stored
'''


def calculate_psnr(images, path1, path2, file_name):
    with open(file_name, "w") as psnr_file:
        psnr_writer = csv.writer(psnr_file)
        for key, values in images.items():
            # print(key, values)
            name = values[0] + '-' + key + '.png'
            img1 = np.array(Image.open(path1 + name).convert('RGB'))
            img2 = np.array(Image.open(path2 + name).convert('RGB'))
            name = values[1] + '-' + key + '.png'
            img3 = np.array(Image.open(path1 + name).convert('RGB'))
            img4 = np.array(Image.open(path2 + name).convert('RGB'))

            psnr1 = psnr(img1, img2)
            psnr2 = psnr(img3, img4)

            psnr_ = (psnr1 + psnr2) / 2
            psnr_writer.writerow([key, psnr_])


mseloss = torch.nn.MSELoss()


def main():
    # ssim as loss functino
    inpainting_ssim = '/Users/tl/Downloads/Inpainting_withSSIM/'
    inpainting_ssim_recon = inpainting_ssim + '/recon/'
    inpainting_ssim_real = inpainting_ssim + '/real/'

    calculate_psnr(imageList(inpainting_ssim_recon), inpainting_ssim_recon, inpainting_ssim_real,
                   "evaluation/inpainting_ssim_psnr.csv")
    calculate_ssim(imageList(inpainting_ssim_recon), inpainting_ssim_recon, inpainting_ssim_real,
                   "evaluation/inpainting_ssim_ssim.csv")

    # loss function only mse
    inpainting_no_ssim = '/Users/tl/Downloads/Inpainting_withoutSSIM/'
    inpainting_no_ssim_recon = inpainting_no_ssim + '/recon/'
    inpainting_no_ssim_real = inpainting_no_ssim + '/real/'
    calculate_psnr(imageList(inpainting_no_ssim_recon), inpainting_no_ssim_recon, inpainting_no_ssim_real,
                   "evaluation/inpainting_no_ssim_psnr.csv")
    calculate_ssim(imageList(inpainting_no_ssim_recon), inpainting_no_ssim_recon, inpainting_no_ssim_real,
                   "evaluation/inpainting_no_ssim_ssim.csv")

    # deblur: ssim 100L1
    deblur_ssim = '/Users/tl/Downloads/Deblur_SSIM/'
    deblur_ssim_real = deblur_ssim + '/real/'
    deblur_ssim_deblur = deblur_ssim + '/deblur/'
    calculate_psnr(imageList(deblur_ssim_deblur), deblur_ssim_deblur, deblur_ssim_real,
                   "evaluation/deblur_ssim_psnr.csv")
    calculate_ssim(imageList(deblur_ssim_deblur), deblur_ssim_deblur, deblur_ssim_real,
                   "evaluation/deblur_ssim_ssim.csv")

    # deblur 100L1 no ssim
    deblur_no_ssim = '/Users/tl/Downloads/Deblur_withoutSSIM/'
    deblur_no_ssim_real = deblur_no_ssim + '/real/'
    deblur_no_ssim_deblur = deblur_no_ssim + '/deblur/'
    calculate_psnr(imageList(deblur_no_ssim_deblur), deblur_no_ssim_deblur, deblur_no_ssim_real,
                   "evaluation/deblur_no_ssim_psnr.csv")
    calculate_ssim(imageList(deblur_no_ssim_deblur), deblur_no_ssim_deblur, deblur_no_ssim_real,
                   "evaluation/deblur_no_ssim_ssim.csv")

    # deblur L1 ssim
    deblur_1_ssim = '/Users/tl/Downloads/Deblur_1_SSIM'
    deblur_1_ssim_real = deblur_1_ssim + '/real/'
    deblur_1_ssim_deblur = deblur_1_ssim + '/deblur/'
    calculate_psnr(imageList(deblur_1_ssim_deblur), deblur_1_ssim_deblur, deblur_1_ssim_real,
                   "evaluation/deblur_1_ssim_psnr.csv")
    calculate_ssim(imageList(deblur_1_ssim_deblur), deblur_1_ssim_deblur, deblur_1_ssim_real,
                   "evaluation/deblur_1_ssim_ssim.csv")

    deblur_1_no_ssim = '/Users/tl/Downloads/Deblur_1_withoutSSIM/'
    deblur_1_no_ssim_real = deblur_1_no_ssim + '/real/'
    deblur_1_no_ssim_deblur = deblur_1_no_ssim + '/deblur/'
    calculate_psnr(imageList(deblur_1_no_ssim_deblur), deblur_1_no_ssim_deblur, deblur_1_no_ssim_real,
                   "evaluation/deblur_1_no_ssim_psnr.csv")
    calculate_ssim(imageList(deblur_1_no_ssim_deblur), deblur_1_no_ssim_deblur, deblur_1_no_ssim_real,
                   "evaluation/deblur_1_no_ssim_ssim.csv")


if __name__ == "__main__":
    main()
