#users/bin/env/python3
import cv2
import numpy as np
import math
import edges

def detect_quadrilateral_edges(image, edges):
    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours and approximate a quadrilateral shape
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # Quadrilateral detected
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)  # Draw in green
            return image, approx

    return image, None


def filter_corners_by_distance(corners, min_distance):
    """ Filter corners to ensure they are at least min_distance apart. """
    filtered_corners = []
    for i in range(len(corners)):
        keep = True
        for j in range(len(filtered_corners)):
            if np.linalg.norm(np.array(corners[i]) - np.array(filtered_corners[j])) < min_distance:
                keep = False
                break
        if keep:
            filtered_corners.append(corners[i])
    return filtered_corners

def reorder_points_clockwise(points):
    # Convert points to a NumPy array for convenience
    points = np.array(points)
    
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)
    
    # Compute the angle of each point relative to the centroid
    def angle_from_centroid(point):
        return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
    
    # Sort the points based on their angles in clockwise order
    angles = np.array([angle_from_centroid(point) for point in points])
    sorted_indices = np.argsort(angles)
    
    # Reorder points based on sorted indices
    sorted_points = points[sorted_indices]
    
    return sorted_points

# def filter_outliers(data, threshold):
#     mean = data.mean()
#     for point in data:
#         if(np.linalg.norm(a-b))

def find_geometric_center_of_quadrilateral(edge_image):
    # Find contours from the edge image
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find the quadrilateral
    quadrilateral = None
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the polygon has 4 vertices (a quadrilateral)
        if len(approx) == 4:
            quadrilateral = approx
            break
    
    if quadrilateral is not None:
        # Compute the centroid of the quadrilateral
        M = cv2.moments(quadrilateral)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        else:
            return None
    else:
        return None

def find_geometric_center(contour):
    # Calculate moments for the contour
    M = cv2.moments(contour)

    # Calculate the center (centroid) coordinates
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        return (center_x, center_y)
    else:
        # If the area (m00) is zero, the centroid can't be determined
        return None

# Function to mark the geometric center on the image
def mark_geometric_center(image, center):
    if center is not None:
        # Mark the center on the image with a red dot
        cv2.circle(image, center, 5, (0, 255, 255), -1)  # Red dot
    else:
        print("Geometric center could not be found.")
    return image

def find_intersections(lines):
    intersections = []
    if lines is not None:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                rho1, theta1 = lines[i][0]
                rho2, theta2 = lines[j][0]
                
                # Convert polar coordinates to Cartesian coordinates
                A1 = np.cos(theta1)
                B1 = np.sin(theta1)
                C1 = rho1
                A2 = np.cos(theta2)
                B2 = np.sin(theta2)
                C2 = rho2
                
                # Calculate the intersection point
                determinant = A1 * B2 - A2 * B1
                if determinant != 0:
                    x = (B2 * C1 - B1 * C2) / determinant
                    y = (A1 * C2 - A2 * C1) / determinant
                    intersections.append((int(x), int(y)))
    
    return intersections

def find_centroid(points):
    if len(points) == 0:
        return None
    
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    
    return (int(centroid_x), int(centroid_y))

if __name__ == "__main__":

    img = cv2.imread("0000390-000013037724.jpg")
    cdst = np.copy(img)
    #print(img.shape)
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
    
    edges = cv2.Canny(mask, 50, 200, None, 3)
    corners = cv2.cornerHarris(edges, 2, 13, 0.04)
    corners = cv2.dilate(corners, None)

    # Apply a threshold to identify strong corners
    threshold = 0.01 * corners.max()
    corner_locations = np.argwhere(corners > threshold)

    # Convert to (x, y) format
    corner_locations = [(x[1], x[0]) for x in corner_locations]
    min_distance = 55  # Minimum distance between corners
    filtered_corners = filter_corners_by_distance(corner_locations, min_distance)
    #filtered_corners = filter_outliers(filtered_contours, 100)

    center = find_centroid(filtered_corners)
    print(center)

    # Display the results
    for corner in filtered_corners:
        cv2.circle(img, corner, 5, (0, 0, 255), 2)

    ordered_corners = reorder_points_clockwise(filtered_corners)

    # max_corners = 4  # Maximum number of corners to return
    # quality_level = 0.05  # Quality level parameter for the corner detection
    # min_distance = 10  # Minimum distance between corners

    # # Detect corners using Shi-Tomasi
    # corners = cv2.goodFeaturesToTrack(edges, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)

    # # Convert corners to integer coordinates
    # corners = corners.astype(np.uint8)

    # # Draw the corners on the image
    # for corner in corners:
    #     x, y = corner.ravel()
    #     cv2.circle(img, (x, y), 3, 255, -1)

    # center = find_geometric_center_of_quadrilateral(edges)

    # if center:
    #     print(f'Geometric Center of the Quadrilateral: {center}')
    # else:
    #     print('Quadrilateral not found')

    # # Display the result

    # # Draw the center on the image
    # cv2.drawContours(img, [cv2.boxPoints(cv2.minAreaRect(center))], -1, (0, 255, 0), 2)
    # cv2.circle(img, center, 5, (0, 0, 255), -1)
    # cv2.imshow('Result Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # img, approx = detect_quadrilateral_edges(img, edges)
    # if approx is not None:
    #     center = find_geometric_center(approx)
    #     processed_image = mark_geometric_center(img, center)
    # else:
    #     center = None

    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)

    # # Find intersections of lines
    # intersections = find_intersections(lines)

    # # Compute the centroid of the detected intersections
    # centroid = find_centroid(intersections)

    # if centroid:
    #     print(f'Geometric Center of the Quadrilateral: {centroid}')
    # else:
    #     print('No intersections found')

    # Display the result

    cv2.circle(img, center, 5, (255, 255, 255), -1)
    cv2.imshow('Result Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


