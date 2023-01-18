import random
import cv2
import mediapipe as mp
import json
import numpy as np
import os

mpPose = mp.solutions.pose
pose = mpPose.Pose()
# ORB Detector
orb = cv2.ORB_create()
# feature matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# if json exist, read object from there
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


def resizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def gait(path,video1):
    while True:
        # This is to check whether to break the first loop
        isclosed = 0
        cap1 = cv2.VideoCapture(path + video1)
        cap2 = cv2.VideoCapture(0)
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(40)]

        OBJ1, flag1 = jsonCheck(video1)
        OBJ2 = {}

        i, j, k, s = 0, 0, 0, 0
        while True:
            success1, img1 = cap1.read()
            success2, img2 = cap2.read()
            if success1 and success2:
                if cv2.waitKey(1) == 27:
                    # When esc is pressed isclosed is 1
                    isclosed = 1
                    break

                img1 = cv2.resize(img1, (640, 480))
                imgRGB1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                imgRGB2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                # if json doesn't exist, create object
                if flag1:
                    results1 = pose.process(imgRGB1)
                    OBJ1['pose ' + str(i)] = {
                        ('point ' + str(i)): {'x': item.x, 'y': item.y, 'z': item.z, 'visibility': item.visibility} for
                        (i, item) in
                        enumerate(results1.pose_landmarks.landmark)}
                    i += 1

                results2 = pose.process(imgRGB2)
                OBJ2['pose ' + str(j)] = {
                    ('point ' + str(j)): {'x': item.x, 'y': item.y, 'z': item.z, 'visibility': item.visibility} for
                    (j, item) in
                    enumerate(results2.pose_landmarks.landmark)}
                j += 1

                # draw points by taking coordinates from json
                for m in range(len(OBJ2["pose " + str(s)])):
                    h1, w1, c1 = img1.shape
                    h2, w2, c2 = img2.shape

                    cx1, cy1 = int(OBJ1["pose " + str(s)]["point " + str(m)]["x"] * w1), int(
                        OBJ1["pose " + str(s)]["point " + str(m)]["y"] * h1)
                    cx2, cy2 = int(OBJ2["pose " + str(s)]["point " + str(m)]["x"] * w2), int(
                        OBJ2["pose " + str(s)]["point " + str(m)]["y"] * h2)

                    # cv2.circle(img, (cx, cy), 5, [colors[id][0]+(i+1), colors[id][1]+(i+1), colors[id][2]+(i+1)], cv2.FILLED)
                    cv2.circle(img1, (cx1, cy1), 8, colors[m], cv2.FILLED)
                    cv2.circle(img2, (cx2, cy2), 8, colors[m], cv2.FILLED)
                s += 1
                # merge both videos in one
                vis = np.hstack([img1, img2])
                # vis = np.concatenate((img1, img2), axis=1)
                # detect and compute features
                keypoints_1, descriptors_1 = orb.detectAndCompute(img1, None)
                keypoints_2, descriptors_2 = orb.detectAndCompute(img2, None)
                # draw matches between founded features
                matches = bf.match(descriptors_1, descriptors_2)
                matches = sorted(matches, key=lambda x: x.distance)
                vis = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:100], img2, flags=2)

                # draw lines between points
                for m in range(len(OBJ2["pose " + str(k)])):
                    x0 = int(OBJ1["pose " + str(k)]["point " + str(m)]["x"] * img1.shape[1])
                    y0 = int(OBJ1["pose " + str(k)]["point " + str(m)]["y"] * img1.shape[0])

                    x1 = int(vis.shape[1] / 2 + (OBJ2["pose " + str(k)]["point " + str(m)]["x"] * img2.shape[1]))
                    y1 = int(OBJ2["pose " + str(k)]["point " + str(m)]["y"] * img2.shape[0])

                    cv2.line(vis, (x0, y0), (x1, y1), colors[m], 2)

                k += 1

                # define the screen resulation
                screen_res = 1280, 720
                scale_width = screen_res[0] / vis.shape[1]
                scale_height = screen_res[1] / vis.shape[0]
                scale = min(scale_width, scale_height)

                # resized window width and height
                window_width = int(vis.shape[1] * scale)
                window_height = int(vis.shape[0] * scale)
                resizeWithAspectRatio(vis, window_width, window_height)

                # show in fullscreen
                cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("Image", vis)
                cv2.waitKey(1)

            else:
                break

        # if json doesn't exist, create new one
        if flag1:
            with open(video1 + ".json", "w") as outfile:
                outfile.write(json.dumps(OBJ1))


        # To break the loop if it is closed manually
        if isclosed:
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    video1 = 'IMG_5022.MOV'
    path = 'data/'
    gait(path, video1)
    #if body/hands are not seen in live capture, than mediapipe returns nontype!
    #camera has mirror effects!

