import cv2
import matplotlib.pyplot as plt
import keyboard

# ORB Detector
orb = cv2.ORB_create()

# feature matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# cap = cv2.VideoCapture(0)
video1 = 'IMG_5022.MOV'
video2 = 'IMG_5006.MOV'
cap1 = cv2.VideoCapture('data/' + video1)
cap2 = cv2.VideoCapture('data/' + video2)

while cap1.isOpened():
    # If key was pressed, destroy window
    if keyboard.is_pressed("esc"):
        cv2.destroyWindow("SIFT")
        break

    # read images
    suc, img1 = cap1.read()
    suc, img2 = cap2.read()

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    keypoints_1, descriptors_1 = orb.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(img2, None)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x:x.distance)

    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:500], img2, flags=2)
    cv2.namedWindow("SIFT", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("SIFT", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('SIFT', img3)


    cv2.waitKey(1)


cap1.release()
cap2.release()