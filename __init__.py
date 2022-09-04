import time
from viewer import AndroidViewer
from cv2 import cv2
import numpy as np
import subprocess
import imutils

WINDOW_WIDTH = 400
WINDOW_HEIGHT = int(WINDOW_WIDTH * (20 / 9))  # usual aspect ratio in newer smartphones

WINDOW_POS_X = 0
WINDOW_POS_Y = 0

FRAME_RATE = 56 # 20/4 ok, 26/4 ok, 35/5 myli się ale wykręcił 652, 42/6 tak jak 35/5(może nawet rzadziej trochę się myli), 48/6 źle, 56/8 rzadko się myli ale i tak 7.0/s


def connectWithDevice():
    # kill possibly running adb process and start new
    subprocess.Popen('.\\scrcpy-win64-v1.24\\adb kill-server')
    adb_serv_process = subprocess.Popen('.\\scrcpy-win64-v1.24\\adb -a nodaemon server start')
    time.sleep(3)

    # time.sleep(5)
    device = AndroidViewer(adb_path='.\\scrcpy-win64-v1.24\\adb.exe', ip='127.0.0.1', port=8081,
                           max_width=WINDOW_HEIGHT,
                           max_fps=FRAME_RATE)
    time.sleep(3)

    return device, adb_serv_process


def detectObject(frame, obj_img):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ORB Detector
    orb = cv2.ORB_create()
    frame_kp, frame_desc = orb.detectAndCompute(frame, None)
    obj_kp, obj_desc = orb.detectAndCompute(obj_img, None)

    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(frame_desc, obj_desc)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) == 0:
        return None

    best_match_dist = matches[0].distance
    best_match_loc = frame_kp[matches[0].queryIdx].pt
    # print('retry dist: ', best_match_dist, 'retry loc: ', best_match_loc)

    return {'dist': best_match_dist, 'loc': best_match_loc}


def findGlove(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define limits of violet HSV values
    violet_lower = np.array([133, 100, 100])
    violet_upper = np.array([153, 255, 255])

    # Filter the image and get the mask
    mask = cv2.inRange(hsv, violet_lower, violet_upper)

    # Remove white noise
    kernel = np.ones((10, 10), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Remove small black dots
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # find contours in the thresholded image
    contours = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) > 0:
        contour = contours[0]
        # compute the center of the contour
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    else:
        return None


def main():
    device, adb_serv_process = connectWithDevice()

    (device_w, device_h) = device.resolution
    dev_window_ratio = device_w / WINDOW_WIDTH

    retry_template = cv2.imread('images/walcz.png', cv2.IMREAD_GRAYSCALE)

    cap_box = {
        'y': int(WINDOW_HEIGHT * (2/3)),
        'x': 0,
        'w': WINDOW_WIDTH,
        'h': 150
    }
    dev_box = {
        'y': int(cap_box['y'] * dev_window_ratio),
        'x': 0,
        'w': int(cap_box['w'] * dev_window_ratio),
        'h': int(cap_box['h'] * dev_window_ratio)
    }

    i = 0
    max_calc_time = 0

    while True:
        frames = device.get_next_frames()

        if frames is None:
            continue

        start = time.time()

        i += 1

        if i % 8 == 0:
            frame = frames[-1]
            frame = frame[cap_box['y']:(cap_box['y'] + cap_box['h']), cap_box['x']:(cap_box['x'] + cap_box['w'])]

            res = findGlove(frame)

            if res is not None:
                x = int(res[0] * dev_window_ratio)
                y = int(res[1] * dev_window_ratio)

                # cv2.circle(frame, (res[0], res[1]), 7, (0, 0, 255), -1) #red
                # cv2.imshow('res', frame)
                # cv2.waitKey(1)

                device.tap(x, y)

                calc_time = time.time() - start
                if calc_time > max_calc_time:
                    max_calc_time = calc_time

                print(time.time() - start)
            else:
                retry = detectObject(frame, retry_template)
                if retry is None:
                    continue

                if retry['dist'] < 20:
                    x = int(retry['loc'][0] * dev_window_ratio)
                    y = int(retry['loc'][1] * dev_window_ratio + dev_box['y'])

                    # cv2.circle(frame, (int(retry['loc'][0]), int(retry['loc'][1])), 7, (0, 255, 0), -1) # green
                    # cv2.imshow('res', frame)
                    # cv2.waitKey(1)

                    device.tap(x, y)

                calc_time = time.time() - start
                if calc_time > max_calc_time:
                    max_calc_time = calc_time

        # print(max_calc_time)


main()