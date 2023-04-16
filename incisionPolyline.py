import matplotlib.pyplot as plt
from rightUp import move_right_slant


def polyline_detection(incision, incision_original):
    x_len = len(incision)
    y_len = len(incision[1])

    incision_points = list()  # for the final detection

    y_position = 0  # on which column is the pixel true
    x_position = 0
    possibilities = list()  # if there is another possible way in the net
    tolerance = 6  # the tolerance band
    steps_up = 3  # how many steps am I able to make UP
    steps_down = 3  # how many steps am I able to make DOWN
    up_counter = 0
    down_counter = 0
    up_down = False
    up_down_steps = 0
    x_diff_right_up = 0
    y_diff_right_up = 0
    x_diff_right_down = 0
    y_diff_right_down = 0

    for y in range(y_position, y_len):
        for x in range(x_position, x_len):
            # direction: right
            if incision[x][y]:
                up_counter = 0
                down_counter = 0
                x_position = x
                points = [x, y]
                # if the current point are not far away from the previous ones
                prev_points = list()
                if len(incision_points) > 1:
                    prev_points = incision_points[-1]
                else:
                    prev_points = points  # first run, so this condition does not make sense
                if abs(points[0]-prev_points[0]) > tolerance or abs(points[1]-prev_points[1]) > tolerance:
                    last_positions = possibilities[-1]
                    x = last_positions[0]
                    y = last_positions[1]
                    del possibilities[-1]
                else:
                    incision_points.append(points)
                # if there is another possibility in the path
                if incision[x-1][y]:
                    possibilities.append([x-1, y])
                if x+1 != x_len and incision[x+1][y]:
                    possibilities.append([x+1, y])
                break

            # direction: right up
            elif incision[x-1][y]:
                outs = move_right_slant(x, y, 0, tolerance, incision_points, possibilities, "up", x_len, incision)
                x_position = outs[0]
                incision_points = outs[1]
                possibilities = outs[2]
                break

            elif incision[x-2][y]:
                outs = move_right_slant(x, y, 1, tolerance, incision_points, possibilities, "up", x_len, incision)
                x_position = outs[0]
                incision_points = outs[1]
                possibilities = outs[2]
                break

            # direction: right down
            elif x+1 != x_len and incision[x+1][y]:
                outs = move_right_slant(x, y, 0, tolerance, incision_points, possibilities, "down", x_len, incision)
                x_position = outs[0]
                incision_points = outs[1]
                possibilities = outs[2]
                break
            """
            elif x+2 != x_len and incision[x+2][y]:
                outs = move_right_slant(x, y, 1, tolerance, incision_points, possibilities, "down", x_len, incision)
                x_position = outs[0]
                incision_points = outs[1]
                possibilities = outs[2]
                break
            """
            """
            # direction: up
            elif up_counter < steps_up and incision[x-1][y-1]:
                up_counter += 1  # one step up already made
                down_counter = 0
                y = y-1
                up_down = True
                up_down_steps += 1  # another step in the previous (same) column
                y_position = y
                x_position = x-1
                points = [x-1, y]
                # if the current point are not far away from the previous ones
                prev_points = list()
                if len(incision_points) > 1:
                    prev_points = incision_points[-1]
                else:
                    prev_points = points  # first run, so this condition does not make sense
                if abs(points[0] - prev_points[0]) > tolerance or abs(points[1] - prev_points[1]) > tolerance:
                    last_positions = possibilities[-1]
                    x = last_positions[0]
                    y = last_positions[1]
                    del possibilities[-1]
                else:
                    incision_points.append(points)
                break
            
            # direction: down
            elif up_counter < steps_up and x+1 < x_len and incision[x+1][y-1]:
                up_counter = 0
                down_counter += 1  # one step up already made
                y = y-1
                up_down = True
                up_down_steps += 1  # another step in the previous (same) column
                y_position = y
                x_position = x+1
                points = [x+1, y]
                # if the current point are not far away from the previous ones
                prev_points = list()
                if len(incision_points) > 1:
                    prev_points = incision_points[-1]
                else:
                    prev_points = points  # first run, so this condition does not make sense
                if abs(points[0] - prev_points[0]) > tolerance or abs(points[1] - prev_points[1]) > tolerance:
                    last_positions = possibilities[-1]
                    x = last_positions[0]
                    y = last_positions[1]
                    del possibilities[-1]
                else:
                    incision_points.append(points)
                break
            """
    x_values = list()
    y_values = list()
    for i in range(0, len(incision_points)):
        x_values.append(incision_points[i][0])
        y_values.append(incision_points[i][1])
    plt.subplot(211)
    plt.plot(y_values, x_values)
    plt.xlim([0, y_len])
    plt.ylim([x_len, 0])
    plt.subplot(212)
    plt.imshow(incision_original, cmap="gray")
    plt.show()

    return incision_points
