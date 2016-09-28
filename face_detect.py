import os

import cv2

dataset_root = '/Users/ilia/temp/hipster_dataset/'
raw_image_path = dataset_root + 'hipster_raw'
no_face_path = dataset_root + 'hipster_no_face'
single_cropped_path = dataset_root + 'hipster'
multiple_cropped_path = dataset_root + 'hipster_multiple'

casc_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(casc_path)

vertical_outline = 0.3

file_count = 1

for image_file_name in os.listdir(raw_image_path):

    # ignore hidden files
    if image_file_name.startswith('.'):
        continue

    print(str(file_count) + "\t" + image_file_name)

    image = cv2.imread(raw_image_path + '/' + image_file_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    if len(faces) == 0:
        cv2.imwrite(no_face_path + '/' + image_file_name, image)
    elif len(faces) == 1:
        (x, y, w, h) = faces[0]
        offset = int(h * vertical_outline)
        cropped_image = image[y - offset:y + h + offset, x:x + w]
        cv2.imwrite(single_cropped_path + '/' + image_file_name, cropped_image)
    else:
        face_count = 1
        for (x, y, w, h) in faces:
            offset = int(h * vertical_outline)
            cropped_image = image[y - offset:y + h + offset, x:x + w]
            tmp_file_name = str(file_count) + '_' + str(face_count) + '_' + image_file_name
            cv2.imwrite(multiple_cropped_path + '/' + tmp_file_name, cropped_image)
            face_count += 1

    file_count += 1



# cv2.imshow("Faces found", image)
# cv2.waitKey(0)
