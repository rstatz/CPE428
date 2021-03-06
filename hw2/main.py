import sys
import math
import os.path
import cv2 as cv
import numpy as np

# Video Paths
viddir = ".\\wands\\"
red_path = viddir + "red.mov"
blue_path = viddir + "blue.mov"
redblue_path = viddir + "redblue.mov"
wand_path = viddir + "wand.mov"

# Wand Config
ball_radius = 3 # cm
wand_length = 36 # cm

# Iphone Config
focal_length = 485.82423388827533
principal_cx = 134.875
principal_cy = 239.875

# Video Settings
frame_period = 25

def camera_coord(p, cx = principal_cx, cy = principal_cy, f = focal_length) :
    x = (p[0] - cx)/f
    y = (p[1] - cy)/f
    return (x, y)

def ball_z(r, br = ball_radius, f = focal_length) :
    return (f * br) / r

# X = z((x - cx)/f)
def xy_2D_to_3D(z, p) :
    cp = camera_coord(p)
    return (cp[0]*z, cp[1]*z)

# x = f(X/z) + cx
def xyz_3D_to_2D(X, Y, Z, cx = principal_cx, cy = principal_cy, f = focal_length) :
    x = ((f * X) / Z) + cx
    y = ((f * Y) / Z) + cy
    return (int(x), int(y))

def get_3D_distance(p1, p2) :
    dx = (p1[0] - p2[0])**2
    dy = (p1[1] - p2[1])**2
    dz = (p1[2] - p2[2])**2

    return math.sqrt(dx + dy + dz)

def cv_draw_line_3D(img, p1, p2, color=(0,255,0), thickness=2) :
    p1_2D = xyz_3D_to_2D(p1[0], p1[1], p1[2])
    p2_2D = xyz_3D_to_2D(p2[0], p2[1], p2[2])

    cv.line(img, p1_2D, p2_2D, color, thickness)

def cv_write_text(img, text, coords, color=(0, 255, 0), thickness=1) :
    return cv.putText(img, text, coords, cv.FONT_HERSHEY_SIMPLEX, .8, color, thickness, cv.LINE_AA)

def cv_find_circles(img) :
    # Find Hough Circles
    rows = img.shape[0]
    return cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1.25, rows / 8, param1=250, param2=70, minRadius=0, maxRadius=0)

def cv_draw_cube(img, X, Y, Z, side_length, color=(0,255,0), thickness=1) :
    r = side_length/2

    xyz_tlf = [X - r, Y - r, Z - r]
    xyz_trf = [X + r, Y - r, Z - r]
    xyz_blf = [X - r, Y + r, Z - r]
    xyz_brf = [X + r, Y + r, Z - r]

    xyz_tlb = [X - r, Y - r, Z + r]
    xyz_trb = [X + r, Y - r, Z + r]
    xyz_blb = [X - r, Y + r, Z + r]
    xyz_brb = [X + r, Y + r, Z + r]

    cv_draw_line_3D(img, xyz_tlf, xyz_trf, color, thickness)
    cv_draw_line_3D(img, xyz_trf, xyz_brf, color, thickness)
    cv_draw_line_3D(img, xyz_brf, xyz_blf, color, thickness)
    cv_draw_line_3D(img, xyz_tlf, xyz_blf, color, thickness)

    cv_draw_line_3D(img, xyz_tlb, xyz_trb, color, thickness)
    cv_draw_line_3D(img, xyz_trb, xyz_brb, color, thickness)
    cv_draw_line_3D(img, xyz_brb, xyz_blb, color, thickness)
    cv_draw_line_3D(img, xyz_tlb, xyz_blb, color, thickness)

    cv_draw_line_3D(img, xyz_tlf, xyz_tlb, color, thickness)
    cv_draw_line_3D(img, xyz_trf, xyz_trb, color, thickness)
    cv_draw_line_3D(img, xyz_brf, xyz_brb, color, thickness)
    cv_draw_line_3D(img, xyz_blf, xyz_blb, color, thickness)

    return img

def cv_draw_circles(circles, img_og, calc_distance=False) :
    if circles is None:
        return img_og, None

    circles = np.uint16(np.around(circles))
    circle_centers_3D = []

    # Cycle through circles found in image
    for c in circles[0, :]:
        circle_center = (c[0], c[1])
        radius = c[2]

        if radius > 0:
            # Calculate circle center in 3D
            Z = ball_z(radius)
            X, Y = xy_2D_to_3D(Z, circle_center)

            circle_centers_3D.append([X,Y,Z])

            # Draw circle center
            cv.circle(img_og, circle_center, 1, (0, 100, 100), 3)
            # Draw circle outline
            cv.circle(img_og, circle_center, radius, (255, 0, 255), 3)

            # Write ball distance
            cv_write_text(img_og, str(round(Z, 2)), circle_center)

            # Draw cube
            cv_draw_cube(img_og, X, Y, Z, 2*ball_radius)

    # Calculate Distance
    if calc_distance and len(circle_centers_3D) == 2 :
        dist = get_3D_distance(circle_centers_3D[0], circle_centers_3D[1])
        return img_og, dist

    return img_og, None

def process_video(vpath, calc_distance=False):
    cap = cv.VideoCapture(vpath)

    if not cap.isOpened() :
        print("Error Reading video at: " + str(sys.argv[1]))

    distances = np.array([])

    while True :
        ret, frame = cap.read()

        # Finish on error or end of video
        if not ret :
            break

        # convert to grayscale and blur
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_blur = cv.GaussianBlur(frame_gray, (9,9), 2)

        circles = cv_find_circles(frame_blur)
        frame_circled, dist = cv_draw_circles(circles, frame, calc_distance)

        if dist is not None :
            distances = np.append(distances, dist)

        cv.imshow(sys.argv[1], frame_circled)

        # break on key press
        if cv.waitKey(frame_period) is not -1:
            break

    if calc_distance :
        mean = np.mean(distances, axis=0)
        std = np.std(distances, axis=0)

        print("Wand Distance Accuracy:\n\tmean:{}\n\tstd:{}".format(mean, std))

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

    print(os.path.abspath(sys.argv[1]))
    print(os.path.abspath(wand_path))

    if os.path.abspath(sys.argv[1]) == os.path.abspath(wand_path) :
        return True
    return False

def main():
    bonus = check_args()
    print("Bonus? {}".format(bonus))

    process_video(sys.argv[1], bonus)

if __name__ == "__main__" :
    main()
