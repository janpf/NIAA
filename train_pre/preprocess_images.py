import sys
from pathlib import Path
from shutil import copyfile
from tempfile import NamedTemporaryFile
from typing import Dict, List, Tuple

import numpy as np
import torchvision.transforms as transforms
from imagenet_c import corrupt
from PIL import Image

from train_pre import utils

import gi

gi.require_version("Gegl", "0.4")
from gi.repository import Gegl

Gegl.init()
Gegl.config().props.application_license = "GPL3"  #  this is essential

mapping = dict()

mapping["styles"] = dict()
mapping["styles"]["brightness"] = ["pos", "neg"]
mapping["styles"]["contrast"] = ["pos", "neg"]
mapping["styles"]["saturation"] = ["pos", "neg"]
mapping["styles"]["exposure"] = ["pos", "neg"]
mapping["styles"]["shadows"] = ["pos", "neg"]
mapping["styles"]["highlights"] = ["pos", "neg"]
mapping["styles"]["temperature"] = ["pos", "neg"]

mapping["technical"] = dict()
mapping["technical"]["jpeg_compression"] = ["pos"]
mapping["technical"]["defocus_blur"] = ["pos"]
mapping["technical"]["motion_blur"] = ["pos"]
mapping["technical"]["pixelate"] = ["pos"]
mapping["technical"]["gaussian_noise"] = ["pos"]
mapping["technical"]["impulse_noise"] = ["pos"]

mapping["composition"] = dict()
mapping["composition"]["rotate"] = ["pos", "neg"]
mapping["composition"]["crop_h"] = ["pos", "neg"]
mapping["composition"]["crop_v"] = ["pos", "neg"]
mapping["composition"]["crop_leftcorners"] = ["pos", "neg"]  # TODO diagcrops useful?
mapping["composition"]["crop_rightcorners"] = ["pos", "neg"]
mapping["composition"]["ratio"] = ["pos", "neg"]

mapping["all_changes"] = [("original", 0)]

for type_change in ["styles", "technical", "composition"]:
    mapping[f"{type_change}_changes"] = []
    for k, v in mapping[type_change].items():
        if "pos" in v:
            mapping[f"{type_change}_changes"].extend([(k, i) for i in range(1, 6)])
        if "neg" in v:
            mapping[f"{type_change}_changes"].extend([(k, i) for i in range(-1, -6, -1)])
    mapping["all_changes"].extend(mapping[f"{type_change}_changes"])


class ImageEditor:
    def __init__(self):
        self.style_maps = {
            "brightness": "brightness-contrast",
            "contrast": "brightness-contrast",
            "shadows": "shadows-highlights",
            "highlights": "shadows-highlights",
            "temperature": "color-temperature",
        }
        self.style_intensity_mapping = {
            "brightness_pos": [float(x) for x in np.linspace(0, 1, 6)],
            "brightness_neg": [float(x) for x in np.linspace(0, -1, 6)],
            "contrast_pos": [float(x) for x in np.linspace(1, 2, 6)],
            "contrast_neg": [float(x) for x in np.linspace(1, 0, 6)],
            "shadows_pos": [float(x) for x in np.linspace(1, 100, 6)],
            "shadows_neg": [float(x) for x in np.linspace(1, -100, 6)],
            "highlights_pos": [float(x) for x in np.linspace(1, 100, 6)],
            "highlights_neg": [float(x) for x in np.linspace(1, -100, 6)],
            "exposure_pos": [float(x) for x in np.linspace(0, 10, 6)],
            "exposure_neg": [float(x) for x in np.linspace(1, -10, 6)],
            "temperature_pos": [float(x) for x in np.linspace(6500, 12000, 6)],
            "temperature_neg": [float(x) for x in np.linspace(6500, 1000, 6)],
            "saturation_pos": [float(x) for x in np.linspace(1, 2, 6)],
            "saturation_neg": [float(x) for x in np.linspace(1, 0, 6)],
        }

    # styles
    def _get_style_node(self, parent_node, distortion: str, intensity_level: int):
        if intensity_level <= 0:
            intensity = self.style_intensity_mapping[f"{distortion}_neg"][abs(intensity_level)]
        else:
            intensity = self.style_intensity_mapping[f"{distortion}_pos"][intensity_level]

        return self._get_style_node_real_values(parent_node, distortion, intensity)

    def _get_style_node_real_values(self, parent_node, distortion: str, intensity: float):
        if distortion in self.style_maps:
            gegl_distortion = self.style_maps[distortion]
        else:
            gegl_distortion = distortion

        edit = parent_node.create_child(f"gegl:{gegl_distortion}")  #  FIXME RGB
        if distortion == "temperature":
            distortion = "intended-temperature"
        elif distortion == "saturation":
            distortion = "scale"
        edit.set_property(distortion, intensity)
        return edit

    # technical
    def _jpeg_compression(self, img: Image.Image, intensity: int) -> Image.Image:
        return Image.fromarray(corrupt(np.array(img), corruption_name="jpeg_compression", severity=intensity))

    def _defocus_blur(self, img: Image.Image, intensity: int) -> Image.Image:
        return Image.fromarray(corrupt(np.array(img), corruption_name="defocus_blur", severity=intensity))

    def _motion_blur(self, img: Image.Image, intensity: int) -> Image.Image:  # TODO same angle?
        return Image.fromarray(corrupt(np.array(img), corruption_name="motion_blur", severity=intensity))

    def _pixelate(self, img: Image.Image, severity: int) -> Image.Image:
        previous = img.size
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
        img = img.resize((int(previous[0] * c), int(previous[1] * c)), Image.BOX)
        img = img.resize(previous, Image.BOX)
        return img

    def _gaussian_noise(self, img: Image.Image, intensity: int) -> Image.Image:
        return Image.fromarray(corrupt(np.array(img), corruption_name="gaussian_noise", severity=intensity))

    def _shot_noise(self, img: Image.Image, intensity: int) -> Image.Image:
        return Image.fromarray(corrupt(np.array(img), corruption_name="shot_noise", severity=intensity))

    def _impulse_noise(self, img: Image.Image, intensity: int) -> Image.Image:
        return Image.fromarray(corrupt(np.array(img), corruption_name="impulse_noise", severity=intensity))

    def _speckle_noise(self, img: Image.Image, intensity: int) -> Image.Image:
        return Image.fromarray(corrupt(np.array(img), corruption_name="speckle_noise", severity=intensity))

    # composition
    def _ratio(self, img: Image.Image, intensity: int, max_ratio: int = 5) -> Image.Image:
        img_size = (img.size[1], img.size[0])
        if intensity > 0:
            img_resize = (img_size[0] * (1 + intensity * (1 / max_ratio)), img_size[1])
        else:
            img_resize = (img_size[0], img_size[1] * (1 + -intensity * (1 / max_ratio)))
        img_resize = (round(img_resize[0]), round(img_resize[1]))
        img = transforms.Resize(img_resize)(img)
        return transforms.CenterCrop(img_size)(img)

    def _rotate(self, img: Image.Image, intensity: int, max_rotate: int = 5) -> Image.Image:
        intensity = intensity * 2
        max_rotate = max_rotate * 2
        rotated = img.rotate(intensity, Image.BICUBIC, True)

        w, h = utils.rotatedRectWithMaxArea(img.size[0], img.size[1], max_rotate)

        rotated = transforms.CenterCrop((h, w))(rotated)
        img_resized = transforms.Resize(min(rotated.size))(img)
        return transforms.CenterCrop((img_resized.size[1], img_resized.size[0]))(rotated)

    def _crop(self, img: Image.Image, intensity_h: int = 0, intensity_v: int = 0) -> Image.Image:
        crop_size = img.size

        center_left = round(img.size[0] / 2 - crop_size[0] / 2)
        center_right = round(img.size[0] / 2 + crop_size[0] / 2)
        center_top = round(img.size[1] / 2 - crop_size[1] / 2)
        center_bottom = round(img.size[1] / 2 + crop_size[1] / 2)

        offset_left = round(center_left * (-intensity_h * (0.2)))
        offset_top = round(center_top * (intensity_v * (0.2)))

        center_left -= offset_left
        center_right -= offset_left
        center_top -= offset_top
        center_bottom -= offset_top

        return img.crop((center_left, center_top, center_right, center_bottom))

    def _crop_h(self, img: Image.Image, intensity: int) -> Image.Image:
        return self._crop(img, intensity_h=intensity)

    def _crop_v(self, img: Image.Image, intensity: int) -> Image.Image:
        return self._crop(img, intensity_v=intensity)

    def _crop_leftcorners(self, img: Image.Image, intensity: int) -> Image.Image:
        return self._crop(img, intensity_h=-abs(intensity), intensity_v=intensity)

    def _crop_rightcorners(self, img: Image.Image, intensity: int) -> Image.Image:
        return self._crop(img, intensity_h=abs(intensity), intensity_v=intensity)

    def distort_image(self, distortion: str, intensity: int, img: Image.Image = None, path: str = None):
        return list(
            self.distort_list_image(img=img, path=path, distortion_intens_tuple_list=[(distortion, intensity)]).values()
        )[0]

    def distort_list_image(
        self, distortion_intens_tuple_list: List[Tuple[str, int]], img: Image.Image = None, path: str = None
    ) -> Dict[str, Image.Image]:  # TODO crop orig, rotate orig
        suffix = Path(path).suffix
        if img is None:
            img = Image.open(path)  #  FIXME RGB

        return_dict: Dict[str, Image.Image] = dict()

        ptn = Gegl.Node()
        ptn.set_property("cache-policy", Gegl.CachePolicy.NEVER)

        with NamedTemporaryFile(suffix=suffix) as src_file:
            if path is not None:
                copyfile(path, src_file.name)
            else:
                img.save(src_file.name)  # FIXME img.format

            orig = ptn.create_child("gegl:load")
            orig.set_property("path", src_file.name)
            orig.set_property("cache-policy", Gegl.CachePolicy.NEVER)

            for distortion, intensity in distortion_intens_tuple_list:
                if distortion == "original":
                    return_dict[distortion] = img
                elif hasattr(self, f"_{distortion}"):
                    return_dict[f"{distortion}_{intensity}"] = getattr(self, f"_{distortion}")(img, intensity)
                else:
                    edit = self._get_style_node(ptn, distortion, intensity)
                    out = ptn.create_child("gegl:save")

                    orig.connect_to("output", edit, "input")
                    edit.connect_to("output", out, "input")

                    with NamedTemporaryFile(suffix=suffix) as out_file:
                        out.set_property("path", out_file.name)
                        out.process()
                        return_dict[f"{distortion}_{intensity}"] = Image.open(out_file.name)
        return return_dict

    def pad_square(self, img: Image.Image, min_size: int = 224, fill_color=(0, 0, 0)) -> Image.Image:
        x, y = img.size
        size = max(min_size, x, y)
        new_img = Image.new("RGB", (size, size), fill_color)
        new_img.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_img
