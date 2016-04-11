import cv2


if __name__ == '__main__':
    print("CV2 version: "+cv2.__version__+'\n')
    img = cv2.imread('../images/out.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # sift = cv2.SI
    # kp = sift.detect(gray, None)

    # img = cv2.drawKeypoints(gray, kp)

    cv2.imshow('gray_Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

