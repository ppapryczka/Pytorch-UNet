# leave one out similar to train vs predict from

from sklearn.model_selection import LeaveOneOut
import os
from train import train_net, dir_img
from predict import predict_img, mask_to_image
import torch
from unet import UNet
from PIL import Image

# train
EPOCHS = 1
BATCH_SIZE = 1
LR = 0.1  # don't used, because of ADAM!
SCALE = 0.5
VAL = 10.0

# predict
THRESHOLD = 0.5


if __name__ == "__main__":
    loo = LeaveOneOut()

    # get all files names from folder dir_img
    files = [
        os.path.splitext(file)[0]
        for file in os.listdir(dir_img)
        if not file.startswith(".")
    ]

    print(files)

    # split to examination groups to protect taking image from same patients
    examination_groups = {}
    for f in files:
        exam_idx = f.split("_")[0]
        if exam_idx not in examination_groups:
            examination_groups[exam_idx] = []
        examination_groups[exam_idx].append(f)

    # reduce dict to list
    examination_groups_list = list(examination_groups.values())

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    for train_index, test_index in loo.split(examination_groups_list):
        train_groups = [examination_groups_list[i] for i in train_index]
        test_groups = [examination_groups_list[i] for i in test_index]

        train_dataset = [item for sublist in train_groups for item in sublist]
        test_dataset = [f"{item}.png" for sublist in test_groups for item in sublist]

        print(test_dataset)

        net = UNet(n_channels=3, n_classes=1, bilinear=True)

        # train net on reduced dataset
        net = train_net(
            net=net,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            device=device,
            img_scale=SCALE,
            val_percent=50.0 / 100,
            images_ids=train_dataset,
        )

        # predict img
        for i, fn in enumerate(test_dataset):
            print(dir_img, fn)
            file_name = os.path.join(dir_img, fn)
            img = Image.open(file_name)
            mask = predict_img(
                net=net,
                full_img=img,
                scale_factor=SCALE,
                out_threshold=THRESHOLD,
                device=device,
            )
            # save result
            result = mask_to_image(mask)
            result.save(fn)
