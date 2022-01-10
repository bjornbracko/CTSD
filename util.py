import cv2
import mediapipe as mp
import math


def check_signal(result, pose):
    left = False
    right = False

    # check left
    l_wrist = result.landmark[pose.PoseLandmark.LEFT_WRIST]
    if l_wrist.x > 0.75:
        left = True

    # check right
    r_wrist = result.landmark[pose.PoseLandmark.RIGHT_WRIST]
    if r_wrist.x < 0.25:
        right = True

    if left is True and right is True:
        left = False
        right = False

    return (left, right)


def check_signal_video(result, pose, cl, cr):
    left = False
    right = False

    # check left
    l_wrist = result.landmark[pose.PoseLandmark.LEFT_WRIST]
    if l_wrist.x > 0.75:
        left = True
        cl += 1
        cr = 0

    # check right
    r_wrist = result.landmark[pose.PoseLandmark.RIGHT_WRIST]
    if r_wrist.x < 0.25:
        right = True
        cr += 1
        cl = 0

    if left is True and right is True:
        left = False
        right = False

    if left is False and right is False:
        cl = 0
        cr = 0

    return (left, right, cl, cr)