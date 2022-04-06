"""
DeblurDataset
generate own Dataset
3 channels: RGB mode
input: root dir
output: blur image, real image
"""
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DeblurDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files_list = os.listdir(os.path.join(self.root_dir, "blur/"))

    # implement __len__() and __getitem__() methods to use torch.utils.data.DataLoader later
    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, index):
        img_file = self.files_list[index]

        blur_img_path = os.path.join(os.path.join(self.root_dir, "blur/"), img_file)
        real_img_path = os.path.join(os.path.join(self.root_dir, "real/"), img_file)

        # Load the input image
        blur_image = Image.open(blur_img_path).convert('RGB')  # .resize([256, 256])
        real_image = Image.open(real_img_path).convert('RGB')  # .resize([256, 256])
        # RuntimeError: stack expects each tensor to be equal size, but got [3, 256, 256] at entry 0 and
        # [1, 256, 256] at entry 15 加上convert RGB使所有的都变成3通道

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        blur = transform(blur_image)
        real = transform(real_image)

        return blur, real


# train_dataset = DeblurDataset('dataset/Places/train')
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# print(len(train_dataset))   # Attention: len = total_num/batch_size 150
# for x, y in train_loader:
#     print(y.shape)   # torch.Size([16, 3, 256, 256])
# print(train_loader)


"""
MaskDataset
generate own Dataset
3 channels: RGB mode
input: root dir
output: central missing image, real image
"""


class MaskDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files_list = os.listdir(os.path.join(self.root_dir, "missing/"))

    # implement __len__() and __getitem__() methods to use torch.utils.data.DataLoader later
    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, index):
        img_file = self.files_list[index]

        missing_img_path = os.path.join(os.path.join(self.root_dir, "missing/"), img_file)
        real_img_path = os.path.join(os.path.join(self.root_dir, "real/"), img_file)

        # Load the input image
        missing_image = Image.open(missing_img_path).convert('RGB')  # .resize([256, 256])
        real_image = Image.open(real_img_path).convert('RGB')  # .resize([256, 256])

        # RuntimeError: stack expects each tensor to be equal size, but got [3, 256, 256] at entry 0 and
        # [1, 256, 256] at entry 15 加上convert RGB使所有的都变成3通道

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        missing = transform(missing_image)
        real = transform(real_image)

        return missing, real

# train_dataset = MaskDataset('dataset/CelebA/train')
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
# print(len(train_loader))  # Attention: len = total_num/batch_size
# for x, y in enumerate(train_loader):
#     print(x)  # torch.Size([16, 3, 256, 256])
#     input_image = y[0]  # mask image
#     target_image = y[1]  # clear image
#     real_cpu = target_image
#     real_center_cpu = real_cpu[:, :, int(256 / 4):int(256 / 4) + int(256 / 2),
#                       int(256 / 4):int(256 / 4) + int(256 / 2)]
#     print(real_center_cpu.shape)   # [16, 3, 128, 128]
#     batch_size = real_cpu.size(0)
#     print(batch_size)
