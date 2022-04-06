"""
run config before train the model
make the directory in this project
"""

import os

try:
    os.mkdir("checkpoint")
    os.mkdir("checkpoint/deblur")
    os.mkdir("checkpoint/inpainting")
    os.mkdir("result")
    os.mkdir("result/deblur/")
    os.mkdir("result/deblur/real")
    os.mkdir("result/deblur/blur")
    os.mkdir("result/deblur/deblur")
    os.mkdir("result/inpainting")
    os.mkdir("result/inpainting/real")
    os.mkdir("result/inpainting/cropped")
    os.mkdir("result/inpainting/recon")
except Exception as e:
    print(e)

