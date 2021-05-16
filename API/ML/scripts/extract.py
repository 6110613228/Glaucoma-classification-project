import cv2 as cv
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


def get_roi(img):

    src = img.copy()

    crop = []

    g = cv.split(img)[1]

    g = cv.GaussianBlur(g, (15, 15), 0)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    g = ndimage.grey_opening(g, structure=kernel)

    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(g)

    center = (maxLoc[0], maxLoc[1])

    crop.append(img)
    crop.append(cv.circle(src, center, 70, (0, 0, 0), -1))
    # crop return list of [resized imamge, OD removed image]
    return crop

def get_cd_r(image):

    m = 60
    gaus = signal.gaussian(m, std=7)
    stdG = gaus.std()

    b, g, r = cv.split(image)

    pre_g = g - g.mean() - g.std()
    pre_r = r - r.mean() - r.std()
    ghist, b_g = np.histogram(pre_g, 256, [0, 256])
    rhist, b_r = np.histogram(pre_r, 256, [0, 256])

    # Smoothen histogram of preprocessed green, red channel
    smooth_r = np.convolve(rhist, gaus)
    smooth_g = np.convolve(ghist, gaus)

    preprocess_r = []
    for i, x in enumerate(smooth_r):
        x = round(x)
        while x > 0:
            preprocess_r.append(i)
            x -= 1

    preprocess_g = []
    for i, x in enumerate(smooth_g):
        x = round(x)
        while x > 0:
            preprocess_g.append(i)
            x -= 1

    preprocess_r = np.array(preprocess_r)
    #
    # T1
    #
    T1 = preprocess_r.std()
    row, col = r.shape
    disc = np.zeros(r.shape[:2])
    for i in range(row):
        for j in range(col):
            if pre_r[i, j] > T1:
                disc[i, j] = 255
            else:
                disc[i, j] = 0

    preprocess_g = np.array(preprocess_g)
    #
    # T2
    #
    T2 = 2.0 * preprocess_g.std()
    row, col = g.shape
    cup = np.zeros(g.shape[:2])
    for i in range(row):
        for j in range(col):
            if pre_g[i, j] > T2:
                cup[i, j] = 255
            else:
                cup[i, j] = 0
    disc = np.uint8(disc)
    disc = cv.medianBlur(disc, 7)
    disc_m = cv.morphologyEx(
        disc, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    disc_m = cv.morphologyEx(
        disc_m, cv.MORPH_OPEN, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    disc_m = cv.morphologyEx(disc_m, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (11, 11)))
    disc_m = cv.morphologyEx(disc_m, cv.MORPH_OPEN, kernel=cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (11, 11)))
    disc_m = cv.morphologyEx(disc_m, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (21, 21)))

    cup = np.uint8(cup)
    cup = cv.medianBlur(cup, 7)
    cup_m = cv.morphologyEx(
        cup, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    cup_m = cv.morphologyEx(
        cup_m, cv.MORPH_OPEN, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    cup_m = cv.morphologyEx(cup_m, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (11, 11)))
    cup_m = cv.morphologyEx(cup_m, cv.MORPH_OPEN, kernel=cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (11, 11)))
    cup_m = cv.morphologyEx(cup_m, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (21, 21)))
    contours = cv.findContours(disc, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[
        0]  # Getting all possible contours in the segmented image

    largest_area = 0
    disc_circle = contours[0]
    if len(contours) != 0:
        for i in range(len(contours)):
            if len(contours[i]) >= 5:
                # Getting the contour with the largest area
                area = cv.contourArea(contours[i])
                if (area > largest_area):
                    largest_area = area
                    disc_circle = cv.minEnclosingCircle(contours[i])

    # Draw Circle for disc on img
    (x, y), radius = disc_circle
    disc_r = int(radius)

    contours = cv.findContours(cup, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[
        0]  # Getting all possible contours in the segmented image
    largest_area = 0
    cup_circle = contours[0]
    if len(contours) != 0:
        for i in range(len(contours)):
            if len(contours[i]) >= 5:
                # Getting the contour with the largest area
                area = cv.contourArea(contours[i])
                if (area > largest_area):
                    largest_area = area
                    cup_circle = cv.minEnclosingCircle(contours[i])
    # Draw Circle for disc on img
    (x, y), radius = cup_circle
    cup_r = int(radius)

    return [disc_r, cup_r]

def get_exudate(img):

    fundus = cv.resize(img, (800, 615))
    fundus_mask = cv.imread('scripts/mask.bmp')
    fundus_mask = cv.resize(fundus_mask, (800, 615))

    f1 = cv.bitwise_and(fundus[:, :, 0], fundus_mask[:, :, 0])
    f2 = cv.bitwise_and(fundus[:, :, 1], fundus_mask[:, :, 1])
    f3 = cv.bitwise_and(fundus[:, :, 2], fundus_mask[:, :, 2])

    fundus_dash = cv.merge((f1, f2, f3))

    b, g, r = cv.split(fundus_dash)
    gray_scale = cv.cvtColor(fundus_dash, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_fundus = clahe.apply(gray_scale)
    contrast_enhanced_green_fundus = clahe.apply(g)
    bv_image_dash = extract_bv(g)
    bv_image = extract_bv(gray_scale)
    edge_feature_output = edge_pixel_image(gray_scale, bv_image)
    newfin = cv.dilate(edge_feature_output, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (5, 5)), iterations=1)
    edge_candidates = cv.erode(newfin, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (3, 3)), iterations=1)
    edge_candidates = np.uint8(edge_candidates)

    mask = np.ones((615, 800)) * 255
    mask = np.uint8(mask)
    mask = cv.circle(mask, (400, 307), 537, (0, 0, 0), 500)

    exudate = cv.bitwise_and(mask, edge_candidates)

    histogram = plt.hist(exudate.ravel(), 256, [0, 256])
    plt.close()

    return histogram[0][255]

def edge_pixel_image(image, bv_image):
    edge_result = image.copy()
    edge_result = cv.Canny(edge_result, 30, 100)
    i = 0
    j = 0
    while i < image.shape[0]:
        j = 0
        while j < image.shape[1]:
            if edge_result[i, j] == 255 and bv_image[i, j] == 255:
                edge_result[i, j] = 0
            j = j+1
        i = i+1
    newfin = cv.dilate(edge_result, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return newfin

def extract_bv(image):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_green_fundus = clahe.apply(image)
    # applying alternate sequential filtering (3 times closing opening)
    r1 = cv.morphologyEx(contrast_enhanced_green_fundus, cv.MORPH_OPEN,
                         cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), iterations=1)
    R1 = cv.morphologyEx(r1, cv.MORPH_CLOSE, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (5, 5)), iterations=1)
    r2 = cv.morphologyEx(R1, cv.MORPH_OPEN, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (11, 11)), iterations=1)
    R2 = cv.morphologyEx(r2, cv.MORPH_CLOSE, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (11, 11)), iterations=1)
    r3 = cv.morphologyEx(R2, cv.MORPH_OPEN, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (23, 23)), iterations=1)
    R3 = cv.morphologyEx(r3, cv.MORPH_CLOSE, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (23, 23)), iterations=1)
    f4 = cv.subtract(R3, contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)

    # removing very small contours through area parameter noise removal
    ret, f6 = cv.threshold(f5, 15, 255, cv.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv.findContours(
        f6.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt) <= 200:
            cv.drawContours(mask, [cnt], -1, 0, -1)

    im = cv.bitwise_and(f5, f5, mask=mask)
    ret, fin = cv.threshold(im, 15, 255, cv.THRESH_BINARY_INV)
    newfin = cv.erode(fin, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (3, 3)), iterations=1)

    # removing blobs of microaneurysm & unwanted bigger chunks taking in consideration they are not straight lines like blood
    # vessels and also in an interval of area
    fundus_eroded = cv.bitwise_not(newfin)
    xmask = np.ones(image.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv.findContours(
        fundus_eroded.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv.contourArea(cnt) <= 3000 and cv.contourArea(cnt) >= 100:
            shape = "circle"
        else:
            shape = "veins"

        if shape == 'circle':
            cv.drawContours(xmask, [cnt], -1, 0, -1)

    finimage = cv.bitwise_and(fundus_eroded, fundus_eroded, mask=xmask)
    blood_vessels = cv.bitwise_not(finimage)
    dilated = cv.erode(blood_vessels, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (7, 7)), iterations=1)
    blood_vessels_1 = cv.bitwise_not(dilated)

    return blood_vessels_1

def features(image):
    
    features = []
    image = get_roi(image)
    c_r, d_r = get_cd_r(image[0])
    exudates = get_exudate(image[1])
    features.append(c_r/d_r)
    features.append(d_r - c_r)
    features.append(exudates)

    return features
