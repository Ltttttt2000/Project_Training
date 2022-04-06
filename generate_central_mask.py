'''
generate missing part of the images
1/4 part of image was missing: white mask
total time: 0:02:09.738379
'''
import os
from datetime import datetime

from PIL import Image
from torchvision import transforms

# path = '/Users/tl/Downloads/part_train/real'
# missing_path = '/Users/tl/Downloads/part_train/missing'

path = 'dataset/CelebA/train/real'
missing_path = 'dataset/CelebA/train/missing'
imageSize = 256
start_time = datetime.now()
for image in os.listdir(path):

    origin_img = Image.open(os.path.join(path, str(image))).convert('RGB')
    transform = transforms.ToTensor()
    tensor = transform(origin_img)

    tensor.data[:, int(imageSize/4):int(imageSize/4+imageSize/2), int(imageSize/4):int(imageSize/4+imageSize/2)] = -0.01
    images = transforms.ToPILImage()
    missing_img = images(tensor).convert('RGB')
    missing_img.save(os.path.join(missing_path, str(image)), "PNG", quality=100)

end_time = datetime.now()
print("total time: " + str(end_time - start_time))


