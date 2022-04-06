'''
preprocessing: generate blurred images for training set and testing set
use three blur ways: Blur, BoxBlur, GaussianBlur separately
Input: clear
Output: blurred images
time: total time: 0:02:10.682957
total time: 0:02:33.377404
total time: 0:03:28.093488
'''
import os
from PIL import Image, ImageFilter
from datetime import datetime

start_time = datetime.now()

origin_path = 'dataset/CelebA/train/real'
blur_path = 'dataset/CelebA/train/blur'

random = 0
for image in os.listdir(origin_path):
    origin_img = Image.open(os.path.join(origin_path, str(image)))
    choose = random % 3
    random = random + 1
    if choose == 0:
        blur_img = origin_img.filter(ImageFilter.BLUR)
        blur_img.save(os.path.join(blur_path, str(image)), "PNG", quality=100)
    elif choose == 1:
        blur_img = origin_img.filter(ImageFilter.GaussianBlur(10))
        blur_img.save(os.path.join(blur_path, str(image)), "PNG", quality=100)
    if choose == 2:
        blur_img = origin_img.filter(ImageFilter.BoxBlur(10))
        blur_img.save(os.path.join(blur_path, str(image)), "PNG", quality=100)

end_time = datetime.now()
print("total time: " + str(end_time - start_time))