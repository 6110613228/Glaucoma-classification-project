import cv2 as cv
import scipy.ndimage as ndimage

def get_roi(img):
    """
    ROI: Region of Interest
    @param image
    @return [same image, Optic disc removed image]
    """
    src = img.copy()

    crop = []

    g = cv.split(img)[1]

    g = cv.GaussianBlur(g, (15,15), 0)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15,15))
    g = ndimage.grey_opening(g, structure=kernel)

    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(g)

    center = (maxLoc[0], maxLoc[1])

    crop.append(img)
    crop.append(cv.circle(src, center, 70, (0, 0, 0), -1))

    return crop
    
def load_image(path):
    image = cv.imread(path)
    return get_roi(image)