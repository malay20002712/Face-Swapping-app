import cv2
import numpy as np
import dlib
import time
source_image = cv2.imread("srk.jpg")
source_image_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
dest_image = cv2.imread("katrina-kaif.jpg")
dest_image_gray = cv2.cvtColor(dest_image, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(source_image_gray)

land_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
source_face = land_detector(source_image_gray)
for face in source_face:
    landmarks = predictor(source_image_gray, face)
    points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points.append((x, y))
       face_point = np.array(points, np.int32)
       convexhull = cv2.convexHull(face_point)

land_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
source_face = land_detector(source_image_gray)
for face in source_face:
    landmarks = predictor(source_image_gray, face)
    points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points.append((x, y))
       face_point = np.array(points, np.int32)
       convexhull = cv2.convexHull(face_point)

rectangle = cv2.boundingRect(convexhull)
divide_2d = cv2.Subdiv2D(rectangle)
divide_2d.insert(landmarks_points)
split_triangle = divide_2d.getTriangleList()
split_triangle = np.array(split_triangle, dtype=np.int32)
cv2.fillConvexPoly(mask, convexhull, 255)
face_image_1 = cv2.bitwise_and(source_image, source_image, mask=mask)
face_points2 = np.array(points2, np.int32)
convexhull2 = cv2.convexHull(face_points2)
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index
    join_indexes = []
    for edge in triangles:
        first = (edge[0], edge[1])
        second = (edge[2], edge[3])
        third = (edge[4], edge[5])
index_edge1 = np.where((points == first).all(axis=1))
        index_edge1 = extract_index_nparray(index_edge1)
        index_edge2 = np.where((points == pt2).all(axis=1))
        index_edge2 = extract_index_nparray(index_edge2)
        index_edge3 = np.where((points == pt3).all(axis=1))
        index_edge3 = extract_index_nparray(index_edge3)
        if index_edge1 is not None and index_edge2 is not None and index_edge3 is not None:
            triangle = [index_edge1, index_edge2, index_edge3]
            join_indexes.append(triangle)

source_mask = np.zeros_like(source_image_gray)
new_face = np.zeros_like(dest_image)
for index in indexes_triangles:
    tri_one = points[index[0]]
    tri_two = points[index[1]]
    tri_three = points[index[2]]
    triangle1 = np.array([tri_one, tri_two, tri_three], np.int32)
    first_rect = cv2.boundingRect(triangle1)
    (x, y, w, h) = first_rect
    cropped_triangle = source_image[y: y + h, x: x + w]
    cropped_tr1_mask = np.zeros((h, w), np.uint8)
    pts = np.array([[tri_one[0] - x, tri_one[1] - y],
                       [tri_two[0] - x, tri_two[1] - y],
                       [tri_three[0] - x, tri_three[1] - y]], np.int32)
    cv2.fillConvexPoly(cropped_tr1_mask, pts, 255)
    cv2.line(source_mask, tri_one, tri_two, 255)
    cv2.line(source_mask, tri_two, tri_three, 255)
    cv2.line(source_mask, tri_one, tri_three, 255)

tri2_one = points2[index[0]]
    tri2_two = points2[index[1]]
    tri2_three = points2[index[2]]
    triangle2 = np.array([tri2_one, tri2_two, tri2_three], np.int32)
    second_rect = cv2.boundingRect(triangle2)
    (x, y, w, h) = second_rect
    cropped = np.zeros((h, w), np.uint8)
    points2 = np.array([[tri2_one[0] - x, tri2_one[1] - y],
                        [tri2_two[0] - x, tri2_two[1] - y],
                        [tri2_three[0] - x, tri2_three[1] - y]], np.int32)
    cv2.fillConvexPoly(cropped, points2, 255)

    points = np.float32(points)
    points2 = np.float32(points2)
    transform = cv2.getAffineTransform(points, points2)
    warping = cv2.warpAffine(cropped_triangle, transform, (w, h))
    warping = cv2.bitwise_and(warping, warping, mask=cropped)

ht, wt, filters = dest_image.shape
dest_face = np.zeros((ht, wt, filters), np.uint8)
 facial_area = dest_face[y: y + h, x: x + w]
    facial_area_gray = cv2.cvtColor(facial_area, cv2.COLOR_BGR2GRAY)
    _,triangle_mask = cv2.threshold(facial_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warping = cv2.bitwise_and(warping, warping, mask=triangle_mask)
facial_area = cv2.add(facial_area, warping)
    dest_face[y: y + h, x: x + w] = facial_area

final_mask = np.zeros_like(dest_image_gray)
head_mask = cv2.fillConvexPoly(final_mask, convexhull2, 255)
final_mask = cv2.bitwise_not(head_mask)
combine = cv2.bitwise_and(dest_image, dest_image, mask=final_mask)
output = cv2.add(combine, dest_face)

(x, y, w, h) = cv2.boundingRect(convexhull2)
seamless= (int((x + x + w) / 2), int((y + y + h) / 2))
seamlessclone = cv2.seamlessClone(output, dest_image, head_mask, seamless, cv2.NORMAL_CLONE)
cv2.imshow("seamlessclone", seamlessclone)
cv2.waitKey(0)
cv2.destroyAllWindows()