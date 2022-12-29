import random
import cv2
import mediapipe as mp
import time
import json
import numpy as np
import os
import keyboard

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

video1 = 'IMG_5022.MOV'
video2 = 'IMG_5006.MOV'
cap1 = cv2.VideoCapture('data/' + video1)
cap2 = cv2.VideoCapture('data/' + video2)
pTime = 0
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(40)]

#if json exist, read object from there
def jsonCheck(video):
    if os.path.isfile(video + '.json') and os.access(video + '.json', os.R_OK):
        # checks if file exists
        with open(video + '.json', "r") as f:
            OBJ = json.load(f)
        flag = 0
        print("JSON for video " + video + " already exists")
    else:
        OBJ = dict()
        flag = 1
        print("JSON for video " + video + " doesnt exist so we created new one")

    return OBJ, flag


OBJ1, flag1 = jsonCheck(video1)
OBJ2, flag2 = jsonCheck(video2)
i, j, k= 0 ,0 ,0
# sift
sift = cv2.SIFT_create()

# feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
while True:

    # If key was pressed, destroy window
    if keyboard.is_pressed("esc"):
        cv2.destroyWindow("Image")
        break

    success1, img1 = cap1.read()
    success2, img2 = cap2.read()
    if not success1 or not success2:
        break

    imgRGB1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # results1 = pose.process(imgRGB1)

    imgRGB2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # results2 = pose.process(imgRGB2)

    #if json doesn exist, create object
    # if flag1:
    #     OBJ1['pose ' + str(i)] = {
    #         ('point ' + str(i)): {'x': item.x, 'y': item.y, 'z': item.z, 'visibility': item.visibility} for (i, item) in
    #         enumerate(results1.pose_landmarks.landmark)}
    #     i += 1
    #
    # if flag2:
    #     OBJ2['pose ' + str(j)] = {
    #         ('point ' + str(j)): {'x': item.x, 'y': item.y, 'z': item.z, 'visibility': item.visibility} for (j, item) in
    #         enumerate(results2.pose_landmarks.landmark)}
    #     j += 1
    #
    # #draw landmarks
    # mpDraw.draw_landmarks(img1, results1.pose_landmarks, mpPose.POSE_CONNECTIONS)
    # mpDraw.draw_landmarks(img2, results2.pose_landmarks, mpPose.POSE_CONNECTIONS)

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    #draw points
    # for id, (lm1, lm2) in enumerate(zip(results1.pose_landmarks.landmark, results2.pose_landmarks.landmark)):
    #     h1, w1, c1 = img1.shape
    #     h2, w2, c2 = img2.shape
    #
    #     cx1, cy1 = int(lm1.x * w1), int(lm1.y * h1)
    #     cx2, cy2 = int(lm2.x * w2), int(lm2.y * h2)
    #
    #     # cv2.circle(img, (cx, cy), 5, [colors[id][0]+(i+1), colors[id][1]+(i+1), colors[id][2]+(i+1)], cv2.FILLED)
    #     cv2.circle(img1, (cx1, cy1), 8, colors[id], cv2.FILLED)
    #     cv2.circle(img2, (cx2, cy2), 8, colors[id], cv2.FILLED)

    #resize videos
    img11 = cv2.flip(img1, 0)
    img11 = cv2.flip(img11, 1)

    img22 = cv2.flip(img2, 0)
    img22 = cv2.flip(img22, 1)

    #merge both videos in one
    # vis = np.concatenate((img11, img22), axis=1)
    vis = cv2.drawMatches(img11, keypoints_1, img22, keypoints_2, matches[:300], img22, flags=2)
    #draw lines between points
    # for m in range(len(OBJ2["pose " + str(k)])):
    #     x0 = int((vis.shape[1] - img11.shape[1]) - (OBJ1["pose " + str(k)]["point " + str(m)]["x"] * img11.shape[1]))
    #     y0 = vis.shape[0] - int((OBJ1["pose " + str(k)]["point " + str(m)]["y"] * img11.shape[0]))
    #
    #     x1 = int(vis.shape[1] - OBJ2["pose " + str(k)]["point " + str(m)]["x"] * img22.shape[1])
    #     y1 = int(vis.shape[0] - (OBJ2["pose " + str(k)]["point " + str(m)]["y"] * img22.shape[0]))
    #     cv2.line(vis, (x0, y0), (x1, y1), colors[m], 5)
    # k+= 1

    #show in fullscreen
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Image", vis)
    cv2.waitKey(1)

#if json doesnt exist, create new one
if flag1:
    with open(video1 + ".json", "w") as outfile:
        outfile.write(json.dumps(OBJ1))

if flag2:
    with open(video2 + ".json", "w") as outfile:
        outfile.write(json.dumps(OBJ2))
