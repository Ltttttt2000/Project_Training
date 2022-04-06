import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from dataset import DeblurDataset  # 读取数据集
from generator import Deblur_G
from discriminator import Deblur_D
from torch.utils.data import DataLoader
from tqdm import tqdm  # 进度条
from skimage.metrics import structural_similarity as sk_cpt_ssim


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='dataset/Places/train', help='path to Places dataset')
parser.add_argument('--checkpoint', default='checkpoint/deblur/', help='path to checkpoint for Places dataset')

parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--Epoch', type=int, default=100, help='number of epochs to train for')  # 25
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

parser.add_argument('--gen', default='checkpoint/deblur/P_gen-50.pth', help="path to generator (to continue training)")
parser.add_argument('--dis', default='checkpoint/deblur/P_dis-50.pth', help="path to discriminator (to continue training)")

parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", help='')
opt = parser.parse_args()
print(opt.device)


def main():
    # initialize the discriminator, generator and their optimizers
    netD = Deblur_D().to(opt.device)
    netG = Deblur_G().to(opt.device)

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # initialize loss objects
    adversarial_loss = nn.BCEWithLogitsLoss()
    content_loss = nn.L1Loss()

    # load previously trained model
    if opt.gen != '':
        netG.load_state_dict(torch.load(opt.gen, map_location=opt.device)['state_dict'])
        netD.load_state_dict(torch.load(opt.dis, map_location=opt.device)['state_dict'])
        optimizerG.load_state_dict(torch.load(opt.gen, map_location=opt.device)['optimizer'])
        optimizerD.load_state_dict(torch.load(opt.dis, map_location=opt.device)['optimizer'])
        start_epoch = torch.load(opt.gen)['epoch']
    else:
        start_epoch = 0

    # Load the training set
    train_dataset = DeblurDataset(root_dir=opt.dataroot)
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)

    print(start_epoch)
    # training loop
    for epoch in range(start_epoch, opt.Epoch):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)  # 只有1个False

        for idx, (x, y) in loop:
            blur, real = x.to(opt.device), y.to(opt.device)
            '''
            将模型中的参数的梯度设为0
            model.zero_grad()
            optimizer.zero_grad() 将模型的参数梯度初始化为0
            loss.backward()   反向传播计算梯度，当网络参量进行反馈时，梯度是累积计算而不是被替换，要对每个batch调用一次zero_grad()
            optimizer.step()  更新所有参数
            '''

            # Train Discriminator
            fake = netG(blur)  # generator生成的假图
            D_real = netD(blur, real)  # hd和ld的判别结果
            D_fake = netD(blur, fake.detach())  # ld和生成的
            D_real_loss = adversarial_loss(D_real, torch.ones_like(D_real))
            D_fake_loss = adversarial_loss(D_fake, torch.zeros_like(D_fake))
            D_loss = D_real_loss + D_fake_loss
            optimizerD.zero_grad()
            D_loss.backward()
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            D_fake = netD(blur, fake)
            G_D_loss = adversarial_loss(D_fake, torch.ones_like(D_fake))
            G_fake_loss = content_loss(fake, real) * 100
            # print(G_fake_loss)
            ssim_list = []
            for i in range(opt.batchSize):
                # TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
                img1 = fake[i].cpu().detach().squeeze().transpose(0, 1).transpose(1, 2).numpy()
                img2 = real[i].cpu().detach().squeeze().transpose(0, 1).transpose(1, 2).numpy()
                # print(img2.shape)
                ssim_list.append(sk_cpt_ssim(img1, img2, channel_axis=2))   # 数组是哪个轴对应通道

            ssim = sum(ssim_list) / len(ssim_list)
            # G_loss = G_fake_loss + G_D_loss
            G_loss = G_fake_loss + G_D_loss + ssim
            optimizerG.zero_grad()
            G_loss.backward()
            optimizerG.step()

            loop.set_description(f'Epoch [{epoch + 1}/{opt.Epoch}]')
            loop.set_postfix(gloss=G_loss.item(), dloss=D_loss.item())

            if idx % 100 == 0:
                save_image(real * 0.5 + 0.5, 'result/deblur/real/' + str(idx) + '-' + str(epoch) + '.png')
                save_image(blur * 0.5 + 0.5, 'result/deblur/blur/' + str(idx) + '-' + str(epoch) + '.png')
                save_image(fake * 0.5 + 0.5, 'result/deblur/deblur/' + str(idx) + '-' + str(epoch) + '.png')
        if epoch % 5 == 0:
            torch.save({'epoch': epoch, 'state_dict': netG.state_dict(), 'optimizer': optimizerG.state_dict()}, opt.checkpoint + 'P_gen-' + str(epoch) + '.pth')
            torch.save({'epoch': epoch, 'state_dict': netD.state_dict(), 'optimizer': optimizerD.state_dict()}, opt.checkpoint + 'P_dis-' + str(epoch) + '.pth')

        # do checkpointing
    torch.save({'epoch': opt.Epoch, 'state_dict': netG.state_dict(), 'optimizer': optimizerG.state_dict()}, opt.checkpoint + 'P_generator.pth')
    torch.save({'epoch': opt.Epoch, 'state_dict': netD.state_dict(), 'optimizer': optimizerD.state_dict()}, opt.checkpoint + 'P_discriminator.pth')


if __name__ == "__main__":
    main()
