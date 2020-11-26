import cv2
from matplotlib import pyplot as plt
import numpy as np


def show_image(img):
    plt.imshow(img)
    plt.show()


def go():
    source_path = 'resource/source.png'
    target_path = 'resource/target.png'

    source_img = cv2.imread(source_path)
    searching_img = cv2.imread(target_path)

    sift = cv2.SIFT_create()
    kp_source, des_source = sift.detectAndCompute(cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY), None)
    kp_searching, des_searching = sift.detectAndCompute(cv2.cvtColor(searching_img, cv2.COLOR_BGR2GRAY), None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_searching, des_source, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.3 * n.distance:
            good.append(m)

    source_points = np.float32([kp_source[m.trainIdx].pt for m in good]).reshape(-1, 2)
    searching_points = np.float32([kp_searching[m.queryIdx].pt for m in good]).reshape(-1, 2)
    _, mask = cv2.findHomography(srcPoints=searching_points,
                                 dstPoints=source_points,
                                 method=cv2.RANSAC,
                                 ransacReprojThreshold=11.0)

    pp = source_points[mask.ravel() == 1]
    x_min = pp[:, 0].min()
    y_min = pp[:, 1].min()
    product_regions = \
        np.array([x_min, y_min, pp[:, 0].max() - x_min, pp[:, 1].max() - y_min]).astype(np.int32).reshape(1, 4)

    def draw_regions(source, res, regions, color=(0, 0, 255), size=4):
        for (x, y, w, h) in regions:
            res[y: y + h, x: x + w] = source[y: y + h, x: x + w]
            cv2.rectangle(res, (x, y), (x + w, y + h), color, size)
        return res

    faded = (source_img * 0.65).astype(np.uint8)

    show_image(draw_regions(source_img, faded.copy(), product_regions, (0, 255, 0)))


if __name__ == '__main__':
    go()
