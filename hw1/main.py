import numpy as np
import cv2 as cv

frames_dir = ".\\frames\\"
jpg = ".jpg"
png = ".png"

img0_name = "000000"
img0_path = frames_dir + img0_name
img0_rpath = img0_path + jpg
img0_wpath = img0_path + png

background_path = frames_dir + 'background' + png

def show_img(name, img, twait) :
    if img is None :
        return None

    cv.imshow(name, img)
    cv.waitKey(twait)

    return img

def show_avg_frame(frames) :
    print('vid shape = ' + str(frames.shape))

    avg_frame = np.mean(frames, axis=0)
    avg_frame = avg_frame.astype('uint8')

    show_img('avg', avg_frame, 0)

    cv.destroyAllWindows()

    return avg_frame

def part1() :
    img0 = cv.imread(img0_rpath)

    print('img size:' + str(img0.shape))
    print('img data:\n' + str(img0))

    img0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)

    show_img('img0', img0, 0)
    cv.destroyAllWindows()

    cv.imwrite(img0_wpath, img0)

def part2():
    cap = cv.VideoCapture(frames_dir + '%06d' + jpg)

    frames = []

    # compile and show frames
    while True :
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

        show_img('vid', frame, 100)

    cap.release()
    cv.destroyAllWindows()

    frames = np.array(frames)
    avg_frame = show_avg_frame(frames)

    cv.imwrite(background_path, avg_frame)

def part3() :
    img0 = cv.imread(img0_wpath, cv.IMREAD_GRAYSCALE)
    back = cv.imread(background_path, cv.IMREAD_GRAYSCALE)

    diff_img = cv.absdiff(img0, back)

    thresh = 40 # tune this
    maxval = 255
    th, thresh_img = cv.threshold(diff_img, thresh, maxval, cv.THRESH_BINARY)
    oth, othresh_img = cv.threshold(diff_img, 0, maxval, cv.THRESH_BINARY+cv.THRESH_OTSU)

    cv.imshow('diff', diff_img)
    cv.imshow('thresh', thresh_img)
    cv.imshow('othresh', othresh_img)

    cv.waitKey(0)
    cv.destroyAllWindows()

def bonus() :
    cap = cv.VideoCapture(frames_dir + '%06d' + jpg)
    background = cv.imread(background_path, cv.IMREAD_GRAYSCALE)

    while True :
        ret, frame = cap.read()

        if not ret:
            break

        # Thresholding
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        diff = cv.absdiff(frame_gray, background)
        oth, othresh_img = cv.threshold(diff, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Bounding Boxes
        bb_img, contours, hierarchy = cv.findContours(othresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        bb_img = cv.cvtColor(bb_img, cv.COLOR_GRAY2BGR)
        for contour in contours :
            x, y, w, h = cv.boundingRect(contour)
            bb_img = cv.rectangle(bb_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        show_img('vid', bb_img, 100)

    cap.release()
    cv.destroyAllWindows()

def main() :
    part1()
    part2()
    part3()
    bonus()

if __name__ == "__main__" :
    main()
