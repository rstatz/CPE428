import numpy as np
import cv2 as cv

datadir = ".\\data\\"
stones_path = datadir + "stones.png"
input_vid_path = datadir + "input.mov"
overlay_path = datadir + "overlay.png"

def show_img(name, img, twait) :
    if img is None :
        return None

    cv.imshow(name, img)
    cv.waitKey(twait)

    return img

def main():
    stones_img = cv.imread(stones_path)
    stones_gray = cv.cvtColor(stones_img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    stones_sift = sift.detectAndCompute(stones_gray)

    show_img("stones_sift", stones_sift, 100)

if __name__ == "__main__" :
    main()
