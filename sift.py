import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os
import matplotlib.pyplot as plt


def getHaarcascadePath():
    path = os.path.dirname(os.path.realpath('__file__'))
    cascadePath = path + '/haarcascade_frontalface_default.xml'
    return cascadePath


img1 = cv2.imread('test1.jpg')
img2 = cv2.imread('test3.jpg')

# convert the images from bgr to rgb
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
print(gray.shape)

plt.figure(figsize=(20, 10))
plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
plt.show()

sift = cv2.SIFT_create()
kp = sift.detect(gray, None)

keypoints = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg', img1)

print(keypoints.shape)
plt.figure(figsize=(20, 10))
plt.imshow(keypoints)
plt.title('Keypoints of Image 1 for reference')
plt.show()

# for face detection
face_cascade = cv2.CascadeClassifier(getHaarcascadePath())

# images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

# detect faces in the 2 images
faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
roi_gray = []
roi_color = []

size = gray1.shape

# crop out only the face of the first and second images
for (x, y, w, h) in faces1:
    extra = int(w / 6)
    x1 = max(0, x - extra)
    y1 = max(0, y - extra)
    x2 = min(size[1], x1 + 2 * extra + w)
    y2 = min(size[0], y1 + 2 * extra + w)

    img1 = cv2.rectangle(img1, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 4)
    roi_gray.append(gray1[y1:y2, x1:x2])
    roi_color.append(img1[y1:y2, x1:x2])

if len(faces1) == 0:
    roi_gray.append(gray1)
    roi_color.append(img1)

size = gray2.shape
for (x, y, w, h) in faces2:
    extra = int(w / 6)
    x1 = max(0, x - extra)
    y1 = max(0, y - extra)
    x2 = min(size[1], x1 + 2 * extra + w)
    y2 = min(size[0], y1 + 2 * extra + w)

    img2 = cv2.rectangle(img2, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 4)
    roi_gray.append(gray2[y1:y2, x1:x2])
    roi_color.append(img2[y1:y2, x1:x2])

if len(faces2) == 0:
    roi_gray.append(gray2)
    roi_color.append(img2)

# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# roi_color=cv2.cvtColor(roi_color,cv2.COLOR_BGR2RGB)

# plot the cropped out grayscale images of the originals
plt.figure(figsize=(20, 10))
plt.imshow(roi_gray[0], cmap='gray', vmin=0, vmax=255)
plt.title('ROI of image 1')
plt.show()

plt.figure(figsize=(20, 10))
plt.imshow(roi_gray[1], cmap='gray', vmin=0, vmax=255)
plt.title('ROI of image 2')
plt.show()

kp1, des1 = sift.detectAndCompute(roi_gray[0], None)
kp2, des2 = sift.detectAndCompute(roi_gray[1], None)

# create a bruteforce matcher
bf = cv2.BFMatcher()

# Match descriptors.
# matches = bf.match(des1,des2)
matches = bf.knnMatch(des1, des2, k=2)

# Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)

# Apply ratio test to filter out only the good matches
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

print(len(matches))
print(len(good))

if len(good) >= 15:
    print("It's a Match")
    print(len(good))
else:
    print("Not a Match")
    print(len(good))

# Draw first 10 matches.
# img3=cv2.drawMatches(roi_gray[0],kp1,roi_gray[0],kp2,matches,None,flags=2)

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], img2, flags=2)
plt.imshow(img3),plt.show()

# img1 = cv2.imread('test1.jpg',0)
# img2 = cv2.imread('test4.jpg',0)
#
# sift = cv2.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
#
# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)
#
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
#
# # cv2.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None, flags=2)
#
# plt.imshow(img3),plt.show()
