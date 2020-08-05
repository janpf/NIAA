# some image destroy algorithms are from here: https://codegolf.stackexchange.com/questions/35005/rearrange-pixels-in-image-so-it-cant-be-recognized-and-then-get-it-back
import random

import numpy as np
from PIL import Image, ImageFilter

from tempfile import SpooledTemporaryFile
from pathlib import Path


def jpeg_compress(in_path: str, compression_quality: int) -> Image.Image:
    img: Image.Image = Image.open(in_path)
    img = img.convert("RGB")
    tmp_file = SpooledTemporaryFile(suffix=Path(in_path).suffix)

    img.save(tmp_file, quality=compression_quality)

    return Image.open(tmp_file)


def reverse_slices(in_path: str, count: int, mode: str = "vertical") -> Image.Image:
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
        top = img.crop(box_origin)
        out_img.paste(top, box_destination)

    return out_img


def shuffle_lines(in_path: str, mode: str = "horizontal") -> Image.Image:
    img: Image.Image = Image.open(in_path)
    img = img.convert("RGB")
    if mode == "vertical":
        img = img.rotate(90)
    img_np = np.array(img)

    np.random.shuffle(img_np)

    out_img: Image.Image = Image.fromarray(img_np)
    if mode == "vertical":
        out_img = out_img.rotate(-90)

    return out_img


def random_swap(in_path: str, distance: int) -> Image.Image:
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

    return img


def rotate_blocks(in_path: str, max_size: int) -> Image.Image:
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

    return img


def blur(in_path: int, radius: int) -> Image.Image:
    img: Image.Image = Image.open(in_path)
    img = img.convert("RGB")

    img = img.filter(ImageFilter.GaussianBlur(radius))

    return img
