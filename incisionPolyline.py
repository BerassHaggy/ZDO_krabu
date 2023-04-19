import matplotlib.pyplot as plt
from rightUp import move_right_slant
from moveRight import move_right


def polyline_detection(incision, incision_original, image):
    x_len = len(incision)
    y_len = len(incision[1])

    incision_points = list()  # for the final detection

    y_position = 0  # on which column is the pixel true
    x_position = 0
    possibilities = list()  # if there is another possible way in the net
    tolerance = 6  # the tolerance band
    incision_continue = True
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
        if incision_continue:
            for x in range(x_position, x_len):
                # print(str(x) + ", " + str(y))
                # direction: right
                if incision[x][y]:
                    outs = move_right(x, y, tolerance, incision_points, possibilities, x_len, incision,
                                      incision_continue, image)
                    x_position = outs[0]
                    incision_points = outs[1]
                    possibilities = outs[2]
                    incision_continue = outs[3]
                    break

                # direction: right up
                elif incision[x-1][y]:
                    outs = move_right_slant(x, y, 0, tolerance, incision_points, possibilities, "up", x_len, incision,
                                            incision_continue, image)
                    x_position = outs[0]
                    incision_points = outs[1]
                    possibilities = outs[2]
                    incision_continue = outs[3]
                    break

                # direction: right up - higher tolerance band
                elif incision[x-2][y]:
                    outs = move_right_slant(x, y, 1, tolerance, incision_points, possibilities, "up", x_len, incision,
                                            incision_continue, image)
                    x_position = outs[0]
                    incision_points = outs[1]
                    possibilities = outs[2]
                    incision_continue = outs[3]
                    break

                # direction: right down
                elif x+1 != x_len and incision[x+1][y]:
                    outs = move_right_slant(x, y, 0, tolerance, incision_points, possibilities, "down", x_len, incision,
                                            incision_continue, image)
                    x_position = outs[0]
                    incision_points = outs[1]
                    possibilities = outs[2]
                    incision_continue = outs[3]
                    break
        else:
            break

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
