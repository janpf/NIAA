import numpy as np
from imagenet_c import corrupt
from PIL import Image, ImageEnhance
from skimage import exposure

mapping = dict()
mapping["styles"] = dict()

mapping["styles"]["brightness"]

mapping["technical"] = dict()
mapping["technical"]["jpeg_compression"] = dict()
mapping["technical"]["jpeg_compression"]["pos"] = [f"jpeg_compression;{i}" for i in range(1, 6)]
mapping["technical"]["defocus_blur"] = dict()
mapping["technical"]["defocus_blur"]["pos"] = [f"defocus_blur;{i}" for i in range(1, 6)]
mapping["technical"]["motion_blur"] = dict()
mapping["technical"]["motion_blur"]["pos"] = [f"motion_blur;{i}" for i in range(1, 6)]
mapping["technical"]["pixelate"] = dict()
mapping["technical"]["pixelate"]["pos"] = [f"pixelate;{i}" for i in range(1, 6)]
mapping["technical"]["gaussian_noise"] = dict()
mapping["technical"]["gaussian_noise"]["pos"] = [f"gaussian_noise;{i}" for i in range(1, 6)]
mapping["technical"]["impulse_noise"] = dict()
mapping["technical"]["impulse_noise"]["pos"] = [f"impulse_noise;{i}" for i in range(1, 6)]

mapping["composition"] = dict()
mapping["composition"]["rotate"] = dict()
mapping["composition"]["rotate"]["neg"] = [f"rotate;{i}" for i in range(-10, 0, 2)]
mapping["composition"]["rotate"]["pos"] = [f"rotate;{i}" for i in range(0, 11, 2) if i != 0]
mapping["composition"]["hcrop"] = dict()
mapping["composition"]["hcrop"]["neg"] = [f"hcrop;{i}" for i in range(-5, 0)]
mapping["composition"]["hcrop"]["pos"] = [f"hcrop;{i}" for i in range(0, 6) if i != 0]
mapping["composition"]["vcrop"] = dict()
mapping["composition"]["vcrop"]["neg"] = [f"vcrop;{i}" for i in range(-5, 0)]
mapping["composition"]["vcrop"]["pos"] = [f"vcrop;{i}" for i in range(0, 6) if i != 0]
mapping["composition"]["leftcornerscrop"] = dict()
mapping["composition"]["leftcornerscrop"]["neg"] = [f"leftcornerscrop;{i}" for i in range(-5, 0)]
mapping["composition"]["leftcornerscrop"]["pos"] = [f"leftcornerscrop;{i}" for i in range(0, 6) if i != 0]
mapping["composition"]["rightcornerscrop"] = dict()
mapping["composition"]["rightcornerscrop"]["neg"] = [f"rightcornerscrop;{i}" for i in range(-5, 0)]
mapping["composition"]["rightcornerscrop"]["pos"] = [f"rightcornerscrop;{i}" for i in range(0, 6) if i != 0]
mapping["composition"]["ratio"] = dict()
mapping["composition"]["ratio"]["neg"] = [f"ratio;{i}" for i in range(-5, 0)]
mapping["composition"]["ratio"]["pos"] = [f"ratio;{i}" for i in range(0, 6) if i != 0]

mapping["change_steps"] = dict()
for distortion in ["styles", "technical", "composition"]:
    mapping["change_steps"][distortion] = dict()
    for parameter in mapping[distortion]:
        mapping["change_steps"][distortion][parameter] = dict()
        for polarity in mapping[distortion][parameter]:
            if len(mapping[distortion][parameter][polarity]) > 0:
                mapping["change_steps"][distortion][parameter][polarity] = 1 / len(mapping[distortion][parameter][polarity])

mapping["all_changes"] = ["original"]

mapping["styles_changes"] = []
mapping["technical_changes"] = []
mapping["composition_changes"] = []

for _, v in mapping["styles"].items():
    for _, polarity in v.items():
        mapping["all_changes"].extend(polarity)
        mapping["styles_changes"].extend(polarity)

for _, v in mapping["technical"].items():
    for _, polarity in v.items():
        mapping["all_changes"].extend(polarity)
        mapping["technical_changes"].extend(polarity)

for _, v in mapping["composition"].items():
    for _, polarity in v.items():
        mapping["all_changes"].extend(polarity)
        mapping["composition_changes"].extend(polarity)


class ImageEditor:

    _kelvin_table = {
        1000: (255, 56, 0),
        1500: (255, 109, 0),
        2000: (255, 137, 18),
        2500: (255, 161, 72),
        3000: (255, 180, 107),
        3500: (255, 196, 137),
        4000: (255, 209, 163),
        4500: (255, 219, 186),
        5000: (255, 228, 206),
        5500: (255, 236, 224),
        6000: (255, 243, 239),
        6500: (255, 249, 253),
        7000: (245, 243, 255),
        7500: (235, 238, 255),
        8000: (227, 233, 255),
        8500: (220, 229, 255),
        9000: (214, 225, 255),
        9500: (208, 222, 255),
        10000: (204, 219, 255),
    }

    # highlights/shadows: https://dsp.stackexchange.com/questions/3387/applying-photoshops-shadow-highlight-correction-using-standard-image-proces ??
    # tint: https://stackoverflow.com/a/32587217/6388328

    # styles
    def _brightness(self, img: Image.Image, intensity: float) -> Image.Image:
        return ImageEnhance.Brightness(img).enhance(intensity)

    def _contrast(self, img: Image.Image, intensity: float) -> Image.Image:
        return ImageEnhance.Contrast(img).enhance(intensity)

    def _saturation(self, img: Image.Image, intensity: float) -> Image.Image:
        return ImageEnhance.Color(img).enhance(intensity)

    def _exposure(self, img: Image.Image, intensity: float) -> Image.Image:
        return Image.fromarray(exposure.adjust_log(np.array(img), intensity))

    def _temperature(self, img: Image.Image, intensity: int) -> Image.Image:
        # https://stackoverflow.com/a/11888449/6388328
        r, g, b = self._kelvin_table[intensity]
        matrix = (r / 255.0, 0.0, 0.0, 0.0, 0.0, g / 255.0, 0.0, 0.0, 0.0, 0.0, b / 255.0, 0.0)
        return img.convert("RGB", matrix)

    # technical
    def _jpeg(self, img: Image.Image, intensity: int) -> Image.Image:
        return corrupt(np.array(img), corruption_name="jpeg compression", severity=intensit)

    # composition

    def edit_image(self, img: Image.Image, parameter: str, intensity: float) -> Image.Image:
        return getattr(self, f"_{parameter}")(img, intensity)
