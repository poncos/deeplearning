
import numpy as np
import math


class Dimension2d:
    width = 0
    height = 0

    def __init__(self, width, height):
        self.width = width
        self.height = height


def convolve_2d(image, kernel):
    hi = len(image)
    hk = len(kernel)
    wi = len(image[0]) if hi > 0 else 0
    wk = len(kernel[0]) if hk > 0 else 0
    hr = int(((hi - hk) / hk) + 1)
    wr = int(((wi - wk) / wk) + 1)

    result = [[0] * wr for y in range(hr)]
    # print("RESULT: ", result)
    ri = 0
    for i in range(0, hi, hk):
        rj = 0
        for j in range(0, wi, wk):
            for k in range(hk):
                for t in range(wk):
                    # print("calculating index [%s,%s]" % (ri, rj))
                    # print("Indices i=%s, k=%s, j=%s, t=%s" % (j,k,j,t))
                    # print("result[%s][%s]=%s" % (ri, rj, result[ri][rj]) )
                    result[ri][rj] += image[i+k][j+t] * kernel[k][t]
                    # print("result[%s][%s] += %s * %s = %s" % (ri, rj, image[i + k][j + t], kernel[k][t],
                    #   result[ri][rj]))
            # print("END VALUE")
            rj += 1
        ri += 1

    # print("Result: ", result)
    return result


def convolve_1d_rgb(image, dimension2d, targetdimension):
    # millis_before = int(round(time.time() * 1000))
    kernel_w = dimension2d.width // targetdimension.width
    kernel_h = dimension2d.height // targetdimension.height

    result = [[[0, 0, 0] for _ in range(targetdimension.height)] for _ in range(targetdimension.width)]

    kernel_window_h = [i for i in range(0, dimension2d.height, kernel_h)][0:targetdimension.height]
    kernel_window_w = [i for i in range(0, dimension2d.width, kernel_w)][0:targetdimension.width]

    kernel_window_h.append(dimension2d.height)
    kernel_window_w.append(dimension2d.width)
    # print("Kernel %sx%s" % (kernel_w, kernel_h))
    result_x = 0
    result_y = 0

    if kernel_window_w[-1] - kernel_window_w[-2] > kernel_w*2:
        kernel_window_w[-2] = kernel_window_w[-1]-(kernel_w+2)
        i = 3
        while kernel_window_w[-i+1] - kernel_window_w[-i] > kernel_w:
            kernel_window_w[-i] = kernel_window_w[-i+1]-(kernel_w+2)
            i += 1
        pass

    if kernel_window_h[-1] - kernel_window_h[-2] > kernel_h*2:
        kernel_window_h[-2] = kernel_window_h[-1]-(kernel_h+2)
        i = 3
        while kernel_window_h[-i+1] - kernel_window_h[-i] > kernel_h:
            kernel_window_h[-i] = kernel_window_h[-i+1]-(kernel_h+2)
            i += 1
        pass
    # millis_after = int(round(time.time() * 1000))
    # print("Delay 1: ", millis_after-millis_before)
    # print("Kernel window w with length : ", len(kernel_window_w), " = ", kernel_window_w)
    # print("Kernel window h with length : ", len(kernel_window_h), " = ", kernel_window_h)
    # millis_before = int(round(time.time() * 1000))
    for y in zip(kernel_window_h, kernel_window_h[1:]):
        kernel_y = y[1] - y[0]
        # print("kernel_window_h zip:", y, ", ", kernel_y)
        for x in zip(kernel_window_w, kernel_window_w[1:]):
            kernel_x = x[1] - x[0]
            # print("kernel_window_x zip:", x, ", ", kernel_x)
            # print("x: %s, y: %s, resultx: %s, resulty: %s" % (x, y, result_x, result_y))
            factor = kernel_y * kernel_x
            for dy in range(kernel_y):
                y_offset_1d = dimension2d.width * (y[0] + dy)
                for dx in range(kernel_x):
                    offset_1d = (y_offset_1d + x[0] + dx) * 3
                    # print(offset_1d)
                    r = image[offset_1d]
                    g = image[offset_1d + 1]
                    b = image[offset_1d + 2]
                    # print("r=%s, g=%s, b=%s" % (r, g, b))
                    result[result_x][result_y][0] += r
                    result[result_x][result_y][1] += g
                    result[result_x][result_y][2] += b
            result[result_x][result_y][0] //= factor
            result[result_x][result_y][1] //= factor
            result[result_x][result_y][2] //= factor

            result_x += 1
        result_y += 1
        result_x = 0
    # millis_after = int(round(time.time() * 1000))
    # print("Delay 2: ", millis_after-millis_before)
    return result


def convolve_1d_rgb_(image, dimension2d, targetdimension):

    # current_dimension = dimension2d.width
    # steps = 0
    # while current_dimension >= targetdimension.width:
    #     current_dimension /= 2
    #     steps += 1

    # print("Steps: ", steps)
    # current_dimension = dimension2d.height
    # steps = 0
    # while current_dimension >= targetdimension.height:
    #    current_dimension /= 2
    #    steps += 1
    min_dimension = dimension2d.width if dimension2d.width < dimension2d.height else dimension2d.height
    print("Min dimension: ", min_dimension)
    # t = dimension2d.width/targetdimension.width
    # steps_w = int(math.log(t, 2))
    # print("Step: ", steps_w)

    # t = dimension2d.height / targetdimension.height
    # steps_h = int(math.log(t, 2))
    # print("Step: ", steps_h)

    t = min_dimension / targetdimension.height
    steps = int(math.log(t, 2))
    print("Step: ", steps)

    kernel_size = int(math.pow(2, steps))
    print("Kernel size: ", kernel_size)

    result_dimension = int(min_dimension/kernel_size)
    result = [[[0, 0, 0] for _ in range(result_dimension)] for _ in range(result_dimension)]
    factor = kernel_size*kernel_size
    result_x = 0
    result_y = 0

    print("Result dimension: ", result_dimension)

    for y in range(0, min_dimension-kernel_size, kernel_size):
        for x in range(0, min_dimension-kernel_size, kernel_size):
            for dy in range(kernel_size):
                y_offset_1d = dimension2d.width * (y + dy)
                for dx in range(kernel_size):
                    offset_1d = (y_offset_1d+x+dx)*3
                    # print(offset_1d)
                    r = image[offset_1d]
                    g = image[offset_1d+1]
                    b = image[offset_1d+2]
                    # print("r=%s, g=%s, b=%s" % (r, g, b))
                    result[result_x][result_y][0] += r
                    result[result_x][result_y][1] += g
                    result[result_x][result_y][2] += b
            result[result_x][result_y][0] //= factor
            result[result_x][result_y][1] //= factor
            result[result_x][result_y][2] //= factor

            result_x += 1
        result_y += 1
        result_x = 0
    return result


def convolve_1d_rgb_backup(image, kernel, dimension2d):

    hk = len(kernel)
    wk = len(kernel[0]) if hk > 0 else 0

    hr = int(((dimension2d.height - hk) / hk) + 1)
    wr = int(((dimension2d.width - wk) / wk) + 1)

    result = [[[0, 0, 0]] * wr for _ in range(hr)]
    for j in range(0, dimension2d.height-hk+1, hk):
        for i in range(0, dimension2d.width*3-(wk*3), 3*wk):
            for k in range(hk):
                r = np.array([int(x) for x in image[dimension2d.width*(k+j)+i:dimension2d.width*(k+j)+i+wk*3:3]])
                g = np.array([int(x) for x in image[dimension2d.width*(k+j)+i+1:dimension2d.width*(k+j)+i+1+wk*3:3]])
                b = np.array([int(x) for x in image[dimension2d.width*(k+j)+i+2:dimension2d.width*(k+j)+i+2+wk*3:3]])

                result[int(j/hk)][int((i/3)/wk)][0] = np.sum(r*kernel[k])
                result[int(j / hk)][int((i / 3) / wk)][1] = np.sum(g * kernel[k])
                result[int(j / hk)][int((i / 3) / wk)][2] = np.sum(b * kernel[k])
                # print(j, ",", i, "=", int(j/hk), ",", int((i/3)/wk))

    return result



