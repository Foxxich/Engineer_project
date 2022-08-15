import os
import cv2
import matplotlib.pyplot as plt


def cascade_path():
    path = os.path.dirname(os.path.realpath('__file__'))
    return path + '/haarcascade_frontalface_default.xml'


def prepare_images(image1, image2):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return img1, img2, gray1, gray2


def get_key_points(gray, image):
    plt.figure(figsize=(20, 10))
    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    plt.show()

    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)

    keypoints = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints.jpg', image)

    print(keypoints.shape)
    plt.figure(figsize=(20, 10))
    plt.imshow(keypoints)
    plt.title('Keypoints of Image 1 for reference')
    plt.show()


def crop(image, roi_color, roi_gray, faces, gray, size):
    for (x, y, w, h) in faces:
        extra = int(w / 6)
        x1 = max(0, x - extra)
        y1 = max(0, y - extra)
        x2 = min(size[1], x1 + 2 * extra + w)
        y2 = min(size[0], y1 + 2 * extra + w)

        image = cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 4)
        roi_gray.append(gray[y1:y2, x1:x2])
        roi_color.append(image[y1:y2, x1:x2])
    return image, roi_color, roi_gray


def plot_cropped(roi_gray):
    # plot the cropped out grayscale images of the originals
    plt.figure(figsize=(20, 10))
    plt.imshow(roi_gray[0], cmap='gray', vmin=0, vmax=255)
    plt.title('ROI of image 1')
    plt.show()
    plt.figure(figsize=(20, 10))
    plt.imshow(roi_gray[1], cmap='gray', vmin=0, vmax=255)
    plt.title('ROI of image 2')
    plt.show()


def final_statistics(img1, img2, kp1, kp2, des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], img2, flags=2)
    plt.imshow(img3), plt.show()


def is_match(good):
    if len(good) >= 15:
        print("It's a Match")
        print(len(good))
    else:
        print("Not a Match")
        print(len(good))


def comparison(image1, image2):
    image1, image2, gray1, gray2 = prepare_images(image1, image2)
    get_key_points(gray1, image1)
    sift = cv2.SIFT_create()

    # for face detection
    face_cascade = cv2.CascadeClassifier(cascade_path())

    # detect faces in the images
    faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
    faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
    roi_gray = []
    roi_color = []

    size1 = gray1.shape
    size2 = gray2.shape

    # crop out only the face of the first and second images
    image1, roi_color, roi_gray = crop(image1, roi_color, roi_gray, faces1, gray1, size1)
    image2, roi_color, roi_gray = crop(image2, roi_color, roi_gray, faces2, gray2, size2)

    if len(faces1) == 0:
        roi_gray.append(gray1)
        roi_color.append(image1)
    if len(faces2) == 0:
        roi_gray.append(gray2)
        roi_color.append(image2)

    plot_cropped(roi_gray)

    kp1, des1 = sift.detectAndCompute(roi_gray[0], None)
    kp2, des2 = sift.detectAndCompute(roi_gray[1], None)

    # create a bruteforce matcher
    bf = cv2.BFMatcher()
    # Match descriptors.
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter out only the good matches
    is_matching = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            is_matching.append([m])

    print(len(matches))
    print(len(is_matching))
    is_match(is_matching)
    # feature matching
    final_statistics(image1, image2, kp1, kp2, des1, des2)


def main():
    img1 = cv2.imread('1.jpg')
    img2 = cv2.imread('2.jpg')
    comparison(img1, img2)


if __name__ == "__main__":
    main()

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
