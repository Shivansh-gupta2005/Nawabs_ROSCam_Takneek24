#users/bin/env/python3
import cv2
import numpy as np

if __name__ == "__main__":

    img = cv2.imread("0000019-000000603288.jpg")
    print(img.shape)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,150])
    higher_white = np.array([255,30,255])

    mask = cv2.inRange(hsv, lower_white, higher_white)
    denoising_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, denoising_kernel, iterations=2)


    # depth_map = cv2.imread("0000001-000000000000.png")
    # lower_depth = np.array([0,0,0])
    # higher_depth = np.array([1,1,1])
    # mask_depth = cv2.inRange(depth_map, lower_depth, higher_depth)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours_depth, _ = cv2.findContours(mask_depth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_area = 100

    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > valid_area]
    # filtered_contours_depth = [cnt for cnt in contours_depth if cv2.contourArea(cnt) > valid_area]


    cv2.drawContours(img, filtered_contours, -1, (0,255,0),2)
    # cv2.drawContours(img, filtered_contours_depth, -1, (0,255,0),2)



    cv2.imshow("vid",img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
