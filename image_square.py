import os

import cv2

dataset_root = '/Users/ilia/temp/craft_dataset_training/'

image_cropped_path = dataset_root + 'normal_cropped'
image_resized_path = dataset_root + 'normal'

file_count = 1

for image_file_name in os.listdir(image_cropped_path):

    # ignore hidden files
    if image_file_name.startswith('.'):
        continue

    print(str(file_count) + "\t" + image_file_name)

    image = cv2.imread(image_cropped_path + '/' + image_file_name)
    resized_image = cv2.resize(image, (300, 300))

    cv2.imwrite(image_resized_path + '/' + image_file_name, resized_image)

    file_count += 1
