import cv2
import matplotlib.pyplot as plt

import definitons


# This function is used to get path for cascade needed
# for the sift;
def cascade_path():
    path = definitons.root_dir + '\\utils\\cascades\\'
    return path + 'haarcascade_frontalface_default.xml'


# This function is used to color given images to RGB and
# also get gray versions of them;
def prepare_images(image1, image2):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return img1, img2, gray1, gray2


# This function is used mainly to show informative image
# with keypoints we use to define matches later;
def show_key_points(gray, image):
    plt.figure(figsize=(20, 10))
    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    plt.show()

    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)

    keypoints = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    print(keypoints.shape)
    plt.figure(figsize=(20, 10))
    plt.imshow(keypoints)
    plt.title('Keypoints of Image 1 for reference')
    plt.savefig(definitons.root_dir + '\\results\\sift_keypoints.jpg')
    plt.show()


# This function is needed to crop image, the way
# it was implemented needs parameters about current face
# and gray copy of the image;
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


# This function is used to plot the cropped out grayscale images of the originals;
def plot_grayscale_images(roi_gray):
    plt.figure(figsize=(20, 10))
    plt.imshow(roi_gray[0], cmap='gray', vmin=0, vmax=255)
    plt.title('ROI of image 1')
    plt.show()
    plt.figure(figsize=(20, 10))
    plt.imshow(roi_gray[1], cmap='gray', vmin=0, vmax=255)
    plt.title('ROI of image 2')
    plt.show()


# This function is used to plot final image with comparing both 1st and 2nd images;
def final_statistics(image1, image2, kp1, kp2, des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(image1, kp1, image2, kp2, matches[:50], image2, flags=2)
    plt.imshow(img3), plt.show()


# This function is used to check up percent of matches with set custom delta;
def is_match(good, matches, test_image, original_image):
    delta = 15
    match_percent = len(good) * 100 / len(matches)
    percent_delta = 2.1
    if match_percent >= percent_delta:
        print('There is a match between {} and {}'.format(test_image, original_image))
        print(len(good))
        return True
    else:
        print('There is NO match between {} and {}'.format(test_image, original_image))
        print(len(good))
        return False


def comparison(test_image, original_image):
    image1 = cv2.imread(test_image)
    image2 = cv2.imread(original_image)
    image1, image2, gray1, gray2 = prepare_images(image1, image2)
    show_key_points(gray1, image1)
    sift = cv2.SIFT_create()
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

    plot_grayscale_images(roi_gray)

    kp1, des1 = sift.detectAndCompute(roi_gray[0], None)
    kp2, des2 = sift.detectAndCompute(roi_gray[1], None)

    # create a bruteforce matcher
    bf = cv2.BFMatcher()
    # match descriptors.
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter out only the good matches
    matcher_count = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            matcher_count.append([m])

    final_statistics(image1, image2, kp1, kp2, des1, des2)
    return is_match(matcher_count, matches, test_image, original_image)
