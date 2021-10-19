import os
import random

data_root_dir = "/home/yuqingz/autonomous_driving/exploration/img_ctnet/mmdetection3d/data/waymo/kitti_format"
image_out_dir = "/home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/data/kitti/images/trainval"
label_out_dir = "/home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/data/kitti/training/label_5"
imageset_out_dir = "/home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/data/kitti/ImageSets_waymo"

# move label & image files
for i in range(5):
    print(f"processing session {str(i)}")

    image_dir = f"{data_root_dir}/training/image_{str(i)}"
    img_files = os.listdir(image_dir)
    for img_f in img_files:
        fname = os.path.join(image_dir, img_f)
        out_fname = f"{image_out_dir}/{str(i)}_{img_f}"
        os.system(f"cp {fname} {out_fname}")

    label_dir = f"{data_root_dir}/training/label_{str(i)}"
    label_files = os.listdir(label_dir)
    for label_f in label_files:
        fname = os.path.join(label_dir, label_f)
        out_fname = f"{label_out_dir}/{str(i)}_{label_f}"
        os.system(f"cp {fname} {out_fname}")


# generate train test trainval split
all_label_files = os.listdir(label_out_dir)
all_label_ids = [f.replace(".txt", "") for f in all_label_files]

test_ids = [i for i in all_label_ids if "0_" in i]
trainval_ids = [i for i in all_label_ids if "0_" not in i]

N_train = int(len(trainval_ids) * 0.9)
random.seed(123)
random.shuffle(trainval_ids)
train_ids = sorted(trainval_ids[:N_train])
val_ids = sorted(trainval_ids[N_train:])
trainval_ids = sorted(trainval_ids)

print(len(test_ids))
print(len(trainval_ids))
print(N_train)
print(len(train_ids))
print(len(val_ids))

with open(f"{imageset_out_dir}/test.txt", "w") as fl:
    fl.write("\n".join(test_ids))
fl.close()

with open(f"{imageset_out_dir}/trainval.txt", "w") as fl:
    fl.write("\n".join(trainval_ids))
fl.close()

with open(f"{imageset_out_dir}/train.txt", "w") as fl:
    fl.write("\n".join(train_ids))
fl.close()

with open(f"{imageset_out_dir}/val.txt", "w") as fl:
    fl.write("\n".join(val_ids))
fl.close()
