# code source: https://stackoverflow.com/questions/43391205/add-padding-to-images-to-get-them-into-the-same-shape
from PIL import Image, ImageOps
import os


def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width -
               pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width -
               pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


if __name__ == "__main__":
    img_path = '/home/yuqingz/autonomous_driving/exploration/data/wod2kitti/image_0'
    output_path = '/home/yuqingz/autonomous_driving/exploration/data/wod2kitti_pad/image_0'  # KITTI
    # output_path = '/home/yuqingz/autonomous_driving/exploration/data/wod2kitti_pad_pascal/image_0'
    expected_size = (1280, 384)  # KITTI
    # expected_size = ()

    all_images = os.listdir(img_path)

    for im_path in all_images:
        img = Image.open(img_path + '/' + im_path)
        if img.width != 1920 or img.height != 1280:
            print(img.width, img.height)
        img = resize_with_padding(img, expected_size)
        img.save(output_path + '/' + im_path)
