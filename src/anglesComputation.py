import numpy as np
from intersectionsCalculation import intersectLines


# method for computing the incision and stitches crossings and angles between them
def proces_data(incisions, stitches):
    # incision = coordinates of the detected incision
    # stitches = coordinates of the detected stitches

    ############
    incision_alphas = []
    incision_lines = []

    for incision in incisions:
        for (p_1, p_2) in zip(incision[:-1], incision[1:]):
            p1 = np.array(p_1)
            p2 = np.array(p_2)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dy == 0:
                alpha = 90.0
            elif dx == 0:
                alpha = 0.0
            else:
                alpha = 90 + 180. * np.arctan(dy / dx) / np.pi
            incision_alphas.append(alpha)
            incision_lines.append([p1, p2])

    stitch_alphas = []
    stitch_lines = []

    for stitch in stitches:
        for (p_1, p_2) in zip(stitch[:-1], stitch[1:]):
            p1 = np.array(p_1)
            p2 = np.array(p_2)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dy == 0:
                alpha = 90.0
            elif dx == 0:
                alpha = 180.0
            else:
                alpha = 90 + 180. * np.arctan(dy / dx) / np.pi
            stitch_alphas.append(alpha)
            stitch_lines.append([p1, p2])

    ###############
    # analyze alpha for each pair of line segments
    intersections = []
    intersections_alphas = []
    for (incision_line, incision_alpha) in zip(incision_lines, incision_alphas):
        for (stitch_line, stitch_alpha) in zip(stitch_lines, stitch_alphas):

            p0, p1 = incision_line
            pA, pB = stitch_line
            (xi, yi, valid, r, s) = intersectLines(p0, p1, pA, pB)
            if valid == 1:
                intersections.append([format(xi, ".2f"), format(yi, ".2f")])
                alpha_diff = abs(incision_alpha - stitch_alpha)
                alpha_diff = 180.0 - alpha_diff if alpha_diff > 90.0 else alpha_diff
                # alpha_diff = 90 - alpha_diff
                intersections_alphas.append(format(alpha_diff, ".2f"))

    return intersections, intersections_alphas
