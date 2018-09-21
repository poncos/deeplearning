

class Dimension2d:
    width = 0
    height = 0

    def __init__(self, width, height):
        self.width = width
        self.height = height


def convolve_1d_rgb(image, dimension2d, targetdimension):
    kernel_w = dimension2d.width // targetdimension.width
    kernel_h = dimension2d.height // targetdimension.height

    result = [[[0, 0, 0] for _ in range(targetdimension.height)] for _ in range(targetdimension.width)]

    kernel_window_h = [i for i in range(0, dimension2d.height, kernel_h)][0:targetdimension.height]
    kernel_window_w = [i for i in range(0, dimension2d.width, kernel_w)][0:targetdimension.width]

    kernel_window_h.append(dimension2d.height)
    kernel_window_w.append(dimension2d.width)
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
    for y in zip(kernel_window_h, kernel_window_h[1:]):
        kernel_y = y[1] - y[0]
        for x in zip(kernel_window_w, kernel_window_w[1:]):
            kernel_x = x[1] - x[0]
            factor = kernel_y * kernel_x
            for dy in range(kernel_y):
                y_offset_1d = dimension2d.width * (y[0] + dy)
                for dx in range(kernel_x):
                    offset_1d = (y_offset_1d + x[0] + dx) * 3
                    r = image[offset_1d]
                    g = image[offset_1d + 1]
                    b = image[offset_1d + 2]
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


