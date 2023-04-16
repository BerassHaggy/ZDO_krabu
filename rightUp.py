def move_right_slant(x, y, band, tolerance, incision_points, possibilities, orientation, x_len, incision):
    points = list()
    x_position = 0
    if orientation == "up":
        x_position = x - 1 - band
        points = [x - 1 + band, y]
    else:
        x_position = x + 1 + band
        points = [x + 1 + band, y]
    # if the current point are not far away from the previous ones
    prev_points = list()
    if len(incision_points) > 1:
        prev_points = incision_points[-1]
    else:
        prev_points = points  # first run, so this condition does not make sense
    if abs(points[0] - prev_points[0]) > tolerance or abs(points[1] - prev_points[1]) > tolerance:
        last_positions = possibilities[-1]
        x_diff_right_up = x - last_positions[0]
        y_diff_right_up = y - last_positions[1]
        del possibilities[-1]
    else:
        incision_points.append(points)
    if orientation == "up":
        if x + 1 != x_len and incision[x + 1][y]:
            possibilities.append([x + 1, y])

    return [x_position, incision_points, possibilities]
