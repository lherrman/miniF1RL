import numpy as np
from numba import njit

@njit
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

@njit
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

@njit
def segment_intersection(p1, p2, p3, p4):
    if intersect(p1, p2, p3, p4):
        a1 = p2[1] - p1[1]
        b1 = p1[0] - p2[0]
        c1 = a1 * p1[0] + b1 * p1[1]

        a2 = p4[1] - p3[1]
        b2 = p3[0] - p4[0]
        c2 = a2 * p3[0] + b2 * p3[1]

        det = a1 * b2 - a2 * b1

        if det == 0:
            return None  # Parallel lines
        else:
            x = (b2 * c1 - b1 * c2) / det
            y = (a1 * c2 - a2 * c1) / det
            return np.array([x, y])
    else:
        return None  # No intersection

@njit
def raycast(position, direction, max_distance, track_boundaries):
    direction_length = np.sqrt(direction[0]**2 + direction[1]**2)
    direction = direction / direction_length

    nearest_intersection = None
    nearest_distance = np.inf

    for boundary in track_boundaries:
        for i in range(len(boundary) - 1):
            p1 = boundary[i]
            p2 = boundary[i + 1]

            intersection_point = segment_intersection(position, position + direction * max_distance, p1, p2)
            if intersection_point is not None:
                distance = np.sqrt((intersection_point[0] - position[0])**2 + (intersection_point[1] - position[1])**2)

                if distance < nearest_distance:
                    nearest_intersection = intersection_point
                    nearest_distance = distance

    return nearest_intersection


