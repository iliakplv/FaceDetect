import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
cascPath = 'haarcascade_frontalface_default.xml'

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle and outline around the faces
for (x, y, w, h) in faces:
    vertical_outline = int(h * 0.3)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.rectangle(image, (x, y - vertical_outline), (x + w, y + h + vertical_outline), (0, 0, 255), 2)
    crop_img = image[y - vertical_outline:y + h + vertical_outline, x:x + w]
    cv2.imwrite('cropped.jpeg', crop_img)
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)

# cv2.imshow("Faces found", image)
# cv2.waitKey(0)