import cv2 as cv

def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(img, M, img_size, flags=cv.INTER_NEAREST)  # keep same size as input image

    return warped
