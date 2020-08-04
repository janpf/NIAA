# some image destroy algorithms are from here: https://codegolf.stackexchange.com/questions/35005/rearrange-pixels-in-image-so-it-cant-be-recognized-and-then-get-it-back
import random

import numpy as np
from PIL import Image, ImageFilter


def jpeg_compress(in_path, out_path, compression_quality):
    img: Image.Image = Image.open(in_path)
    img = img.convert("RGB")
    img.save(out_path, quality=compression_quality)


def reverse_slices(in_path, out_path, count, mode="vertical"):
    img: Image.Image = Image.open(in_path)
    img = img.convert("RGB")
    x, y = img.size
    if mode == "vertical":
        cuts = [(x // count) * i for i in range(count)] + [x]
    elif mode == "horizontal":
        cuts = [(y // count) * i for i in range(count)] + [y]
    out_img = Image.new("RGB", (x, y))

    for i in range(count):
        if mode == "vertical":
            box_origin = (cuts[i], 0, cuts[i + 1], y)
            box_destination = (cuts[-(i + 2)], 0, cuts[-(i + 1)], y)
        elif mode == "horizontal":
            box_origin = (0, cuts[i], x, cuts[i + 1])
            box_destination = (0, cuts[-(i + 2)], x, cuts[-(i + 1)])
        print(box_origin, "->", box_destination)
        top = img.crop(box_origin)
        out_img.paste(top, box_destination)

    out_img.save(out_path)


def shuffle_lines(in_path, out_path, mode="horizontal"):
    img: Image.Image = Image.open(in_path)
    img = img.convert("RGB")

    if mode == "vertical":
        img = img.rotate(90)

    img_np = np.array(img)

    np.random.shuffle(img_np)

    out_img: Image.Image = Image.fromarray(img_np)
    if mode == "vertical":
        out_img = out_img.rotate(-90)
    out_img.save(out_path)


def random_swap(in_path, out_path, distance):
    img: Image.Image = Image.open(in_path)
    img = img.convert("RGB")
    x, y = img.size
    count = 3 * (x * y)  # highly empirical
    pixels = img.load()

    for _ in range(count):
        x_origin = random.randrange(x)
        y_origin = random.randrange(y)
        diff = random.randint(-distance, distance)
        x_dest = (x_origin - diff) % x
        y_dest = (y_origin - diff) % y
        pixels[x_dest, y_dest], pixels[x_origin, y_origin] = pixels[x_origin, y_origin], pixels[x_dest, y_dest]  # swapperoni
    img.save(out_path)


def rotate_blocks(in_path, out_path, max_size):
    img: Image.Image = Image.open(in_path)
    img = img.convert("RGB")
    x, y = img.size

    for i in range(2, max_size):
        for xi in range(0, x // i * i, i):
            for yi in range(0, y // i * i, i):
                square_box = (xi, yi, xi + i, yi + i)
                square = img.crop(square_box)
                square = square.rotate(180)
                img.paste(square, square_box)
    for i in reversed(range(2, max_size - 1)):
        for xi in range(0, x // i * i, i):
            for yi in range(0, y // i * i, i):
                square_box = (xi, yi, xi + i, yi + i)
                square = img.crop(square_box)
                square = square.rotate(180)
                img.paste(square, square_box)
    img.save(out_path)


def blur(in_path, out_path, radius):
    img: Image.Image = Image.open(in_path)
    img = img.convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius))
    img.save(out_path)
