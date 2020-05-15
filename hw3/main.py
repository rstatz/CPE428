import cv2 as cv
import numpy as np

img_dir = ".\\images\\"
cat_dir = img_dir + "cat.bmp"
dog_dir = img_dir + "dog.bmp"

ksize = 31

def main():
    cat = cv.imread(cat_dir, cv.IMREAD_COLOR)
    dog = cv.imread(dog_dir, cv.IMREAD_COLOR)

    cat = cat.astype(float) / 255
    dog = dog.astype(float) / 255

    # Build Gaussian Low Pass Filter
    lp_kernel = cv.getGaussianKernel(ksize=ksize,sigma=5,ktype=cv.CV_64F)
    lp_matrix = lp_kernel*lp_kernel.transpose()

    # Build All Pass Filter Kernel
    ap_matrix = np.zeros(shape=(ksize, ksize))
    ap_matrix[int(ksize/2), int(ksize/2)] = 1

    # Build High Pass Filter
    hp_matrix = ap_matrix - lp_matrix

    # Perform Filtration
    cat_hp = cv.filter2D(cat, cv.CV_64F, hp_matrix)
    dog_lp = cv.filter2D(dog, cv.CV_64F, lp_matrix)
    hybrid = cat_hp + dog_lp

    cv.imshow("cat hp", cat_hp)
    cv.imshow("dog lp", dog_lp)

    cv.imshow("hybrid", hybrid)

    cv.waitKey(0)

if __name__ == "__main__" :
    main()
