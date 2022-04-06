"""

"""
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tqdm import tqdm
from generator import Inpainting_G
from discriminator import Inpainting_D
from dataset import MaskDataset
from skimage.metrics import structural_similarity as sk_cpt_ssim


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot_c', default='dataset/CelebA/train', help='path to CelebA dataset')
parser.add_argument('--checkpoint_c', default='checkpoint/In_SSIM/', help='path to checkpoint for Places dataset')
parser.add_argument('--out_c', default='result/In_SSIM/', help='path to train result')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--Epoch', type=int, default=100, help='number of epochs to train for')  # 25
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

parser.add_argument('--gen', default='checkpoint/In_SSIM/I_generator-60.pth', help="path to generator (to continue training)")
parser.add_argument('--dis', default='checkpoint/In_SSIM/I_discriminator-60.pth', help="path to discriminator (to continue training)")

parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", help='')

opt = parser.parse_args()
print(opt.device)

cudnn.benchmark = True  # 让内置的cuddn的auto-tuner自动寻找最合适当前配置的高效算法，来达到优化运行效率的问题。 网络的输入数据维度或类型上变化不大。


def main():
    # custom weights initialization called on generator and discriminator
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    #  initialize the discriminator, generator and their optimizers
    netG = Inpainting_G().to(opt.device)
    netG.apply(weights_init)
    netD = Inpainting_D().to(opt.device)
    netD.apply(weights_init)

    # setup optimizer
    optimizerD = optim.SGD(netD.parameters(), lr=opt.lr)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # setup loss function
    adversarial_loss = nn.BCEWithLogitsLoss().to(opt.device)
    content_loss = nn.MSELoss().to(opt.device)

    if opt.gen != '':
        # load all tensors onto GPU1
        netG.load_state_dict(torch.load(opt.gen, map_location=lambda storage, location: storage)['state_dict'])
        netD.load_state_dict(torch.load(opt.dis, map_location=lambda storage, location: storage)['state_dict'])
        optimizerG.load_state_dict(torch.load(opt.gen, map_location=lambda storage, location: storage)['optimizer'])
        optimizerD.load_state_dict(torch.load(opt.dis, map_location=lambda storage, location: storage)['optimizer'])
        start_epoch = torch.load(opt.gen)['epoch']
    else:
        start_epoch = 0
    # load dataset
    dataset = MaskDataset(opt.dataroot_c)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

    print(start_epoch)
    for epoch in range(int(start_epoch), opt.Epoch):
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)  # 只有1个False
        for index, (x, y) in loop:
            mask = x.to(opt.device)  # mask image
            real = y.to(opt.device)  # clear image   [batch, 3, 256, 256]

            # 获取center部分 [16, 3, 128, 128] 真实的中间部分
            real_center = real[:, :, int(opt.imageSize / 4):int(opt.imageSize / 4) + int(opt.imageSize / 2),
                          int(opt.imageSize / 4):int(opt.imageSize / 4) + int(opt.imageSize / 2)]

            # train discriminator with real (real_label)
            netD.zero_grad()
            output_real = netD(real_center)  # [16, 3, 28, 28]-->[16] judge real center
            errD_real = adversarial_loss(output_real, torch.ones_like(output_real))   # real_label

            # train with fake
            fake_center = netG(mask)
            output_fake = netD(fake_center.detach())
            errD_fake = adversarial_loss(output_fake, torch.zeros_like(output_fake))    # fake_label

            # 只是为了后面的打印内容
            D_x = errD_real.data.mean()
            D_G_z1 = errD_fake.mean()
            # print(D_x, D_G_z1)

            optimizerD.zero_grad()
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            output = netD(fake_center)
            errG_D = adversarial_loss(output, torch.ones_like(output))
            errG_fake = content_loss(fake_center, real_center)

            ssim_list = []
            for i in range(opt.batchSize):
                img1 = fake_center[i].cpu().detach().squeeze().transpose(0, 1).transpose(1, 2).numpy()
                img2 = real_center[i].cpu().detach().squeeze().transpose(0, 1).transpose(1, 2).numpy()
                ssim_list.append(sk_cpt_ssim(img1, img2, channel_axis=2))

            # print(sum(ssim_list)/len(ssim_list))
            ssim = sum(ssim_list)/len(ssim_list)
            # errG = errG_D + errG_fake   # l2 loss weight + errG_D
            errG = errG_D + errG_fake + ssim
            optimizerG.zero_grad()
            errG.backward()
            optimizerG.step()

            loop.set_description(f'Epoch [{epoch + 1}/{opt.Epoch}]')  # [{index + 1}/{len(dataloader)}]
            loop.set_postfix(gloss=errG.item(), dloss=errD.item())

            # IndexError: invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++
            # to convert a 0-dim tensor to a number 修改loss.data[0]为loss.item()
            if index % 100 == 0:
                vutils.save_image(real * 0.5 + 0.5, opt.out_c + 'real/' + str(index) + '-' + str(epoch) + '.png')
                vutils.save_image(mask.data * 0.5 + 0.5, opt.out_c + 'cropped/' + str(index) + '-' + str(epoch) + '.png')
                recon_image = mask.clone()
                recon_image.data[:, :, int(opt.imageSize / 4):int(opt.imageSize / 4 + opt.imageSize / 2),
                int(opt.imageSize / 4):int(opt.imageSize / 4 + opt.imageSize / 2)] = fake_center.data
                vutils.save_image(recon_image.data * 0.5 + 0.5, opt.out_c + 'recon/' + str(index) + '-' + str(epoch) + '.png')
        if epoch % 5 == 0:
            torch.save({'epoch': epoch, 'state_dict': netG.state_dict(), 'optimizer': optimizerG.state_dict()}, opt.checkpoint_c + 'I_generator-' + str(epoch) + '.pth')
            torch.save({'epoch': epoch, 'state_dict': netD.state_dict(), 'optimizer': optimizerD.state_dict()}, opt.checkpoint_c + 'I_discriminator-' + str(epoch) + '.pth')

    # do checkpointing
    torch.save({'epoch': opt.Epoch, 'state_dict': netG.state_dict(), 'optimizer': optimizerG.state_dict()}, opt.checkpoint_c + 'I_generator.pth')
    torch.save({'epoch': opt.Epoch, 'state_dict': netD.state_dict(), 'optimizer': optimizerD.state_dict()}, opt.checkpoint_c + 'I_discriminator.pth')


if __name__ == "__main__":
    main()
