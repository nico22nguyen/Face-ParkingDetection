import cv2 as cv
import numpy as np
import math
import random

######################### PART 1 #########################
def generate_faces():
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    for i in range(10):
        img = cv.imread(f'assets/images/image{i+1}.jpg')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            img_crop = img[y:y+h, x:x+w]
            cv.imshow(f'face_crop {i+1}', img_crop)
            cv.imshow(f'face {i+1}', img)
            cv.imwrite(f'assets/faces/face{i+1}.jpg', img_crop)
    cv.waitKey(0)
    cv.destroyAllWindows()

def test_face_matcher(draw=True):
    matcher = cv.BFMatcher()
    sift = cv.SIFT.create()

    face_num = random.randint(1, 10)
    print(f'Testing matcher using face {face_num}...\n')

    target_face = cv.imread(f'assets/faces/face{face_num}.jpg')
    target_gray = cv.cvtColor(target_face, cv.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(target_gray, None)

    best_match = [0, -1] # format: [best_index, num_matches]
    for img_num in range(10):
        test_image = cv.imread(f'assets/images/image{img_num+1}.jpg')
        test_gray = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(test_gray, None)

        # get matches, filter for good quality matches
        matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        # determine best match so far
        if (len(good)) > best_match[1]:
            best_match = [img_num + 1, len(good)]
        print(f'best match so far: img {best_match[0]}, # good matches: {best_match[1]}')
        
        # show images for this iteration
        if draw:
            matches_img = cv.drawMatchesKnn(target_face, kp1, test_image, kp2, good, None,
                                            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv.imshow('matches', matches_img)
            cv.waitKey(0)
    
    # print results
    print(f'\nBest match was img {best_match[0]} with {best_match[1]} good matches.')
    print(f'TEST {"PASSED" if face_num == best_match[0] else "FAILED"}')
    cv.destroyAllWindows()
    return face_num == best_match[0]

generate_faces()
test_face_matcher()
######################### END PART 1 #########################

######################### PART 2 #########################
def line_is_vertical(line, straightness_threshold=50):
    return abs(line[2] - line[0]) < straightness_threshold

def coalesce_lines(lines, coalesce_radius=40):
    used = []
    coalesced_verts = []
    coalesced_horizontals = []

    for i, line in enumerate(lines):
        if i in used:
            continue

        this_line_is_vertical = line_is_vertical(line)
        group = [line]
        for j, other_line in enumerate(lines):
            if i == j or j in used:
                continue

            # do not mix orientations
            if this_line_is_vertical != line_is_vertical(other_line):
                continue

            # get distances between each endpoint of the two lines
            start_dist = math.dist(line[:2], other_line[:2])
            end_dist = math.dist(line[2:], other_line[2:])
            start_end_dist = math.dist(line[:2], other_line[2:])
            end_start_dist = math.dist(line[2:], other_line[:2])

            endpoint_distances = np.array([start_dist, end_dist, start_end_dist, end_start_dist])

            # if any of the endpoints are close, they are part of the same line, add to group
            if np.any(endpoint_distances < 2 * coalesce_radius):
                group.append(other_line)
                used.append(j)
                
        group = np.array(group)
        group_avg = np.average(group, axis=0).astype(np.int32)

        # all lines in group are combined into one line (max length, average position)
        if this_line_is_vertical:
            group_line = [group_avg[0], np.amin(group[:, 1]), group_avg[2], np.amax(group[:, 3])]
            coalesced_verts.append(group_line)
        else:
            group_line = [np.amin(group[:, 0]), group_avg[1], np.amax(group[:, 2]), group_avg[3]]
            coalesced_horizontals.append(group_line)
        used.append(i)

    return coalesced_verts, coalesced_horizontals

def find_spots(ordered_lines, horizontals):
    ordered = []
    queued = []
    for i, line in enumerate(ordered_lines[:-1]):
        next_line = ordered_lines[i + 1]

        # wrapping point, order existing spots and reset queue
        if line[0] > next_line[0]:
            ordered.extend(queued)
            queued = []
            continue

        # find closest horizontal, this will determine where the y midpoint is to create the spot
        closest_horizontal = -99
        for h in horizontals:
            if abs(h[1] - line[1]) < abs(closest_horizontal - line[1]):
                closest_horizontal = h[1]

        # put a spot at midpoint of upper half and lower half of the two lines
        top_spot = [(line[0] + next_line[0]) // 2, (line[1] + closest_horizontal) // 2]
        bottom_spot = [(line[2] + next_line[2]) // 2, (line[3] + closest_horizontal) // 2]

        ordered.append(top_spot)
        queued.append(bottom_spot)

    ordered.extend(queued)
    return ordered

def orient_lines(lines, straightness_threshold=50):
    for line in lines:
        # vertical -- dx is small
        if line_is_vertical(line, straightness_threshold):
            # oriented upside down ? (bottom first)
            if line[1] > line[3]:
                (line[1], line[3]) = (line[3], line[1])
                (line[0], line[2]) = (line[2], line[0])
        
        # horizontal -- dy is small
        elif abs(line[3] - line[1]) < straightness_threshold:
            # oriented backwards ? (right first)
            if line[0] > line[2]:
                (line[1], line[3]) = (line[3], line[1])
                (line[0], line[2]) = (line[2], line[0])
    return lines

def draw_lines_and_spots(img, lines):
    LINE_COLOR = (0, 0, 200)
    # image to draw lines on
    out_img = img.copy()

    # ensure lines are drawn top to bottom, left to right
    orient_lines(lines)

    # combine duplicate lines into single lines
    filtered_vert, horizontals = coalesce_lines(lines)

    # sort lines top to bottom, left to right within array
    filtered_vert.sort(key=lambda line: line[0])
    filtered_vert.sort(key=lambda line: line[1] // (img.shape[0] // 2)) # sort by y (major key) [fuzzy equality]

    # find spots between lines
    spots = find_spots(filtered_vert, horizontals)

    # draw lines and spots
    SPOT_X_OFFSET = -15
    for line in [*filtered_vert, *horizontals]:
        cv.line(out_img, line[0:2], line[2:4], LINE_COLOR, 2)
    for i, (x, y) in enumerate(spots):
        cv.putText(out_img, str(i + 1), (x + SPOT_X_OFFSET, y), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    print(f'Found {len(spots)} parking spots in the image.')
    return out_img

img = cv.imread(f'assets/parking-lot.jpg')

img_blur = cv.GaussianBlur(img, (5, 5), 0)
img_gray = cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY) 
canny = cv.Canny(img_gray, 50, 225, apertureSize=3)
hough_p = cv.HoughLinesP(canny, 1, np.pi / 180, 75, None, 60, 45)

lines = [line[0] for line in hough_p] 
spots = draw_lines_and_spots(img, lines)

cv.imshow('Original', img)
cv.imshow('Parking Spots', spots)
cv.waitKey(0)
######################### END PART 2 #########################