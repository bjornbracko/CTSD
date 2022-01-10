import cv2
import mediapipe as mp
import math
from os import listdir
import util

DESIRED_HEIGHT = 700
DESIRED_WIDTH = 700

THRESHOLD = 30

def webcam():
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
            else:
                continue

            # Flip the image horizontally for a selfie-view display.
            l, r, cl, cr = res


            ### mogoce kombinacija v movement.cs da pobrise ko prebere in ne napisemo nic ce nic ne zaznamo (ce bo problem uskladiti)

            f = open('statusArm.txt', 'w')
            if l is True or r is True:
                if cl >= THRESHOLD or cr >= THRESHOLD:
                    if l is True:
                        f.write('left')
                        print('left')
                    else:
                        f.write('right')
                        print('right')
            else:
                f.write('none')
            f.close()

            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


if __name__ == '__main__':
    webcam()
