import cv2
import os
import imageio
from PIL import Image
import argparse


def make_gif(img_path, output_path, order=False):
    image_files = os.listdir(img_path)
    image_files = [i for i in image_files if i.endswith(
        '.png') or i.endswith('.jpg')]

    # order by idx
    if order:
        image_idxes = [int("".join([char for char in s if char.isdigit()]))
                       for s in image_files]
        image_files = sorted(
            image_files, key=lambda x: image_idxes[image_files.index(x)])

    images = []
    for fp in image_files:
        # img = cv2.imread(img_path + '/' + fp)
        img = cv2.cvtColor(cv2.imread(img_path + '/' + fp), cv2.COLOR_BGR2RGB)
        images.append(img)

    with imageio.get_writer(output_path, mode="I") as writer:
        for idx, im in enumerate(images):
            # print("Adding frame to GIF file: ", idx + 1)
            if len(images) > 100:
                if 100 <= idx < 200:
                    writer.append_data(im)
            else:
                writer.append_data(im)

    # im = Image.open(output_path + "/det.gif")
    # im.seek(im.tell() + 1)  # loads all frames
    # im.save(output_path + "/det.gif", save_all=True, optimize=True, quality=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str)

    args = parser.parse_args()
    img_path = args.img_path
    if img_path[-1] == "/":
        img_path = img_path[:len(img_path)-1]
    file_name = img_path.split('/')[-1]
    # print("img_path:", img_path, "\nfile_name:", file_name)
    output_path = f'./user_data/output_gif/{file_name}.gif'

    make_gif(img_path, output_path, order=True)


if __name__ == '__main__':
    main()
