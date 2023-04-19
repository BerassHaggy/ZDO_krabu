def move_right(x, y, tolerance, incision_points, possibilities, x_len, incision, incision_continue, image):
    x_position = x
    points = [x, y]
    # if the current point are not far away from the previous ones
    prev_points = list()
    if len(incision_points) > 1:
        prev_points = incision_points[-1]
    else:
        prev_points = points  # first run, so this condition does not make sense
    if abs(points[0] - prev_points[0]) > tolerance or abs(points[1] - prev_points[1]) > tolerance:
        if len(possibilities) > 1:
            last_positions = possibilities[-1]
            x = last_positions[0]
            y = last_positions[1]
            del possibilities[-1]
        else:
            incision_continue = False  # there are no more possibilities
            print("Error in image file: " + image)
    else:
        incision_points.append(points)
    # if there is another possibility in the path
    if incision[x - 1][y]:
        possibilities.append([x - 1, y])
    if x + 1 != x_len and incision[x + 1][y]:
        possibilities.append([x + 1, y])

    return [x_position, incision_points, possibilities, incision_continue]
