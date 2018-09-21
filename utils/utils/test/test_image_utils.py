import unittest

import main.image_utils as iutils
from tkinter import *

from PIL import Image
import time
import os

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

TEST_IMAGE_FILE_NAME = os.path.join(CURRENT_DIR_PATH, '../../../cifar10/resources/test/images/cat3.jpg')


class ImageUtilsTestCase(unittest.TestCase):

    def test_convolve_2d(self):
        image = [[2, 3, 4, 5], [6, 7, 8, 9], [1, 2, 3, 4], [5, 6, 7, 8]]
        kernel = [[0.25, 0.25], [0.25, 0.25]]

        result = iutils.convolve_2d(image, kernel)

        print("Convolution result: ", result)

    def test_convolve_1d(self):
        # img = Image.open("/store/datasets/downloaded_images/animals/cat/cat3.jpg", mode="r")
        img = Image.open(TEST_IMAGE_FILE_NAME, mode="r")
        img_bytes = img.tobytes()
        dimension = iutils.Dimension2d(img.size[0], img.size[1])
        targetdimension = iutils.Dimension2d(250, 250)

        millis_before = int(round(time.time() * 1000))
        result = iutils.convolve_1d_rgb(img_bytes, dimension, targetdimension)
        millis_after = int(round(time.time() * 1000))

        # print("Convolution result: ", result)
        print("Image size ", len(img_bytes))
        print("Image dimension, ", dimension.width, ",", dimension.height)
        print("Image byte: ", img_bytes[0])
        print("Time: ", (millis_after-millis_before), "ms")

        master = Tk()
        canvas = Canvas(master,
                    width=800,
                    height=800)
        canvas.pack()
        photo_image = PhotoImage(width=800, height=800)
        canvas.create_image(0, 0, anchor=NW, image=photo_image)

        for x in range(250):
            for y in range(250):
                r = "%0.2X" % result[x][y][0]
                g = "%0.2X" % result[x][y][1]
                b = "%0.2X" % result[x][y][2]

                photo_image.put("#%s%s%s" % (r, g, b), (x, y))

        mainloop()



if __name__ == '__main__':
    unittest.main()
