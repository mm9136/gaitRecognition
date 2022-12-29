import cv2
import matplotlib.pyplot as plt
import keyboard

# sift
sift = cv2.SIFT_create()

# feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    # If key was pressed, destroy window
    if keyboard.is_pressed("esc"):
        cv2.destroyWindow("Image")
        break

    # read images
    suc, img1 = cap.read()
    img2 = img1
    # img1 = cv2.imread('pencil3.jpg')
    # img2 = cv2.imread('pencil3.jpg')

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:300], img2, flags=2)
    # plt.imshow(img3),plt.show()
    # cv2.putText(img3, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.imshow('SIFT', img3)


    cv2.waitKey(1)


cap.release()