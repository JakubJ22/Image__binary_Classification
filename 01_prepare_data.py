import os
import numpy as np
import shutil

executive_path = os.getcwd()
main_dir = os.path.join(executive_path, 'Data')

CLS_1 = 'Car'
CLS_2 = 'Bike'


TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
DATA_DIR = r'./images'

raw_no_of_files = {}

classes = [CLS_1, CLS_2]
number_of_samples = [
    (dir, len(os.listdir(os.path.join(main_dir, dir)))) for dir in classes]

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

train_dir = os.path.join(DATA_DIR, 'train')
valid_dir = os.path.join(DATA_DIR, 'valid')
test_dir = os.path.join(DATA_DIR, 'test')


# creating folder for every cls and dataset
for i in range(1, len(classes)+1):
    exec(f"train_cls_{i}_dir = os.path.join(train_dir, f'CLS_{i}')")
    exec(f"valid_cls_{i}_dir = os.path.join(valid_dir, f'CLS_{i}')")
    exec(f"test_cls_{i}_dir = os.path.join(test_dir,   f'CLS_{i}')")

for dir in (train_dir, valid_dir, test_dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


train_list = [train_cls_1_dir, train_cls_2_dir]
valid_list = [valid_cls_1_dir, valid_cls_2_dir]
test_list = [test_cls_1_dir, test_cls_2_dir]

for train, valid, test in zip(train_list, valid_list, test_list):
    for dir in (train, valid, test):
        if not os.path.exists(dir):
            os.mkdir(dir)


# Gathering image names
for i, clas in enumerate(classes):
    i += 1
    exec(f"cls_{i}_names = {os.listdir(os.path.join(main_dir, clas))}")

# cls_names = [cls_1_names, cls_2_names, cls_3_names, cls_4_names]
cls_names = [cls_1_names, cls_2_names]

# Image names validation
for name in cls_names:
    name = [name_2 for name_2 in name if name_2.split('.')[1].lower() in [
        'jpg', 'png', 'jpeg']]

# randomize sequences of images
for name in cls_names:
    np.random.shuffle(name)

for clas, name in zip(classes, cls_names):
    print(f"The number of images of {clas} : {len(name)}")

for i, name in enumerate(cls_names):

    exec(f"train_idx_cls_{i+1} = int(TRAIN_RATIO * len({name}))")
    exec(
        f"valid_idx_cls_{i+1} = train_idx_cls_{i+1} + int(VALID_RATIO * len({name}))")

print("\nFiles are being copied to a final destination...\n")


train_idx_list = [train_idx_cls_1, train_idx_cls_2]
valid_idx_list = [valid_idx_cls_1, valid_idx_cls_2]

for train, valid, test, train_idx, valid_idx, clas, cls in zip(train_list, valid_list, test_list, train_idx_list, valid_idx_list, classes, cls_names):
    for i, name in enumerate(cls):
        if i <= train_idx:
            src = os.path.join(main_dir, clas, name)
            dst = os.path.join(train, name)
            shutil.copy(src, dst)

        if train_idx < i <= valid_idx:
            src = os.path.join(main_dir, clas, name)
            dst = os.path.join(valid, name)
            shutil.copy(src, dst)

        if valid_idx < i <= len(cls):
            src = os.path.join(main_dir, clas, name)
            dst = os.path.join(test, name)
            shutil.copy(src, dst)

for i, pack in enumerate(zip(train_list, valid_list, test_list)):

    train, valid, test = pack

    print(
        f"The number of train files of {classes[i]} is {len(os.listdir(train))}")
    print(
        f"The number of validation files of {classes[i]} is {len(os.listdir(valid))}")
    print(
        f"The number of test files of {classes[i]} is {len(os.listdir(test))}\n")
