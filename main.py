import cv2
import mediapipe as mp
import math
from os import listdir
import util

DESIRED_HEIGHT = 700
DESIRED_WIDTH = 700

THRESHOLD = 30

DIR = 'data3'

def resize_and_show(image, res):
    h, w = image.shape[:2]
    l, r = res
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    h, w = img.shape[:2]
    if l is True or r is True:
        mid_x = int(w/2)
        mid_y = int(h/2)
        if l is True:
            end_point = (mid_x + 100, mid_y)
        else:
            end_point = (mid_x - 100, mid_y)
        img = cv2.arrowedLine(img, (mid_x, mid_y), end_point, (255, 0, 0), 5)

    cv2.imshow('img', img)
    cv2.waitKey(0)


def static():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # For static images:
    IMAGE_FILES = listdir(DIR)
    print(IMAGE_FILES)
    BG_COLOR = (192, 192, 192) # gray

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:

      for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(DIR + '/' + file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
          continue

        res = util.check_signal(results.pose_landmarks, mp_pose)

        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        resize_and_show(annotated_image, res)
    cv2.destroyAllWindows()

def webcam():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)

    cl = 0
    cr = 0

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue


            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                res = util.check_signal_video(results.pose_landmarks, mp_pose, cl, cr)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            l, r, cl, cr = res
            h, w = image.shape[:2]
            if l is True or r is True:
                if cl >= THRESHOLD or cr >= THRESHOLD:
                    mid_x = int(w / 2)
                    mid_y = int(h / 2)
                    if l is True:
                        end_point = (mid_x + 100, mid_y)
                    else:
                        end_point = (mid_x - 100, mid_y)
                    image = cv2.arrowedLine(image, (mid_x, mid_y), end_point, (255, 0, 0), 5)
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

def video(file):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture('data_video/' + file)
    cl = 0
    cr = 0
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break


            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                res = util.check_signal_video(results.pose_landmarks, mp_pose, cl, cr)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            l, r, cl, cr = res
            h, w = image.shape[:2]
            if l is True or r is True:
                if cl >= THRESHOLD or cr >= THRESHOLD:
                    mid_x = int(w / 2)
                    mid_y = int(h / 2)
                    if l is True:
                        end_point = (mid_x + 300, mid_y)
                    else:
                        end_point = (mid_x - 300, mid_y)
                    image = cv2.arrowedLine(image, (mid_x, mid_y), end_point, (255, 0, 0), 20)
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == '__main__':
    #static()
    #webcam()
    video('IMG_9976.mov')