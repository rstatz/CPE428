import numpy as np
import cv2 as cv

# File Paths
datadir = ".\\data\\"
stones_path = datadir + "stones.png"
input_vid_path = datadir + "input.mov"
overlay_path = datadir + "overlay.png"

# Config Values
FRAME_PERIOD = 10
RATIO_TEST_THRESHOLD = .7
MIN_MATCH_COUNT = 10

def show_img(name, img, twait) :
    if img is None :
        return None

    cv.imshow(name, img)
    cv.waitKey(twait)

    return img

def sift_match_video(vpath, ipath):
    # Open Video
    cap = cv.VideoCapture(vpath)

    if not cap.isOpened() :
        print("Error Reading video at: " + str(vpath))

    # Open image
    img = cv.imread(ipath)
    if img is None:
        print("Issue with opening " + img_path)
        return None

    # Get Sift features of image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    img_kp, img_des = sift.detectAndCompute(gray, None)

    # Draw Sift Features on Image
    sift_img = cv.drawKeypoints(gray, img_kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_img("stones_sift", sift_img, 1)

    inlier_percentages = []
    bf = cv.BFMatcher()

    # Process Video Frames
    while True :
        ret, frame = cap.read()

        # Finish on error or end of video
        if not ret :
            break

        img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Get sift features of frame
        frame_kp, frame_des = sift.detectAndCompute(img_gray, None)
        #frame_sift = cv.drawKeypoints(img_gray, frame_kp, frame, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Find Matches with ratio test
        matches = bf.knnMatch(img_des, frame_des, k=2)
        good_matches_draw = []
        good_matches = []
        for m,n in matches:
            if m.distance < RATIO_TEST_THRESHOLD * n.distance :
                good_matches_draw.append([m])
                good_matches.append(m)

        # Draw Matches
        # frame_sift_matches = cv.drawMatchesKnn(sift_img, img_kp, frame, frame_kp, good_matches_draw, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv.imshow(vpath, frame_sift_matches)

        # Find homography
        if len(good_matches) > MIN_MATCH_COUNT :
            src_pts = np.float32([img_kp[h.queryIdx].pt for h in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_kp[h.trainIdx].pt for h in good_matches]).reshape(-1, 1, 2)

            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w, d = img.shape
            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts, M)

            hg_img = cv.polylines(frame, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
            # cv.imshow("homography", hg_img)
        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None

        # Find Inliers
        draw_params = dict(matchColor = (0,255,0), singlePointColor=None, matchesMask = matchesMask, flags=2)
        inliers_img = cv.drawMatches(img, img_kp, frame, frame_kp, good_matches, None, **draw_params)
        cv.imshow("inliers", inliers_img)

        # Calculate percent inliers for frame
        p = 0
        for m in matchesMask :
            if m == 1 :
                p+=1
        inlier_percentages.append(p/len(matchesMask))

        # Break on key press
        if cv.waitKey(FRAME_PERIOD) is not -1:
            break

    # Find average number percentage of inliers
    sum=0
    for p in inlier_percentages:
        sum+=p
    avg_percent_inliers = sum/len(inlier_percentages)

    print("Avg % Inliers (threshold={}) : {}".format(RATIO_TEST_THRESHOLD,avg_percent_inliers))

def main():
    sift_match_video(input_vid_path, stones_path)
    cv.destroyAllWindows()

if __name__ == "__main__" :
    main()
