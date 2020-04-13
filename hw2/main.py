import sys
import os.path
import cv2 as cv
import numpy as np

# Video Paths
viddir = ".\\wands\\"
red_path = viddir + "red.mov"
blue_path = viddir + "blue.mov"
redblue_path = viddir + "redblue.mov"
wand_path = viddir + "wand.mov"

#Video Settings
frame_period = 50

def part1(vpath):
    cap = cv.VideoCapture(vpath)

    if not cap.isOpened() :
        print("Error Reading video at: " + str(sys.argv[1]))

    while True :
        ret, frame = cap.read()
        # Finish on error or end of video
        if not ret :
            break

        # convert to grayscale and blur
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_blur = cv.GaussianBlur(frame_gray, (9,9), 2)

        # Hough Circles
        rows = frame_blur.shape[0]
        circles = cv.HoughCircles(frame_blur, cv.HOUGH_GRADIENT, 1, rows/8, param1=100, param2=30, minRadius=1, maxRadius=30)

        frame_circled = frame

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for c in circles[0, :]:
                center = (c[0], c[1])
                # circle center
                frame_circled = cv.circle(frame, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = c[2]
                frame_circled = cv.circle(frame, center, radius, (255, 0, 255), 3)

        cv.imshow(sys.argv[1], frame_circled)

        # break on key press
        if cv.waitKey(frame_period) is not -1:
            break

def usage() :
    print("Usage: python main video_path")
    exit()

def check_args() :
    if len(sys.argv) != 2 :
        usage()
    if not isinstance(sys.argv[1], str) :
        usage()
    if not os.path.isfile(sys.argv[1]) :
        print("Invalid Path")
        exit()

def main():
    check_args()

    part1(sys.argv[1])


if __name__ == "__main__" :
    main()
