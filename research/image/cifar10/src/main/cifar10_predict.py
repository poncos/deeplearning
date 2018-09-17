
import main.cifar10_vgg16_model as cm
import main.cifar10_loader as cl

# import main.image_utils as iutils

from tkinter import *
import os
import tensorflow as tf
import operator

from PIL import Image

NUM_TRAINING_RECORDS = 10000

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

MODEL_FILE_PREFIX = os.path.join(CURRENT_DIR_PATH, '../../resources/model/-980')

print("current dir %s" % CURRENT_DIR_PATH)


def preprocess_image(image_data):
    pass


def cifar10_predict_main():
    img = Image.open("/store/datasets/downloaded_images/animals/cat/cat3.jpg", mode="r")
    img_bytes = img.tobytes()
    width, height = img.size

    master = Tk()

    canvas = Canvas(master,
                    width=width,
                    height=height)
    canvas.pack()
    photo_image = PhotoImage(width=width, height=height)
    canvas.create_image(0, 0, anchor=NW, image=photo_image)
    # img.put("#FF0000", (1, 1))
    # for j in range(height):
    #    for i in range(width):
    #        img.put(img_bytes[j*height+i], (i,j))
    for d in range(0,len(img_bytes),3):
        r = "%0.2X" % img_bytes[d]
        g = "%0.2X" % img_bytes[d+1]
        b = "%0.2X" % img_bytes[d+2]

        y = int((d/3)/width)
        x = int((d/3) % width)
        # print("r=%s, g=%s, b=%s" % (r,g,b))
        photo_image.put("#%s%s%s" % (r, g, b), (x, y))

    mainloop()


if __name__ == '__main__':
    cifar10_predict_main()
