import hashlib
from edit_image import parameter_range

mapping = dict()
mapping["styles"] = dict()

for style in parameter_range.keys():
    if style == "lcontrast":
        continue
    mapping["styles"][style] = dict()
    mapping["styles"][style]["neg"] = [f"{style};{i}" for i in parameter_range[style]["range"] if i < parameter_range[style]["default"]]
    mapping["styles"][style]["pos"] = [f"{style};{i}" for i in parameter_range[style]["range"] if i > parameter_range[style]["default"]]

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
mapping["composition"]["rotate"]["neg"] = [f"rotate;{i}" for i in range(-15, 0, 3)]
mapping["composition"]["rotate"]["pos"] = [f"rotate;{i}" for i in range(0, 16, 3) if i != 0]
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

mapping["style_changes"] = []
mapping["technical_changes"] = []
mapping["composition_changes"] = []

for _, v in mapping["styles"].items():
    for _, polarity in v.items():
        mapping["all_changes"].extend(polarity)
        mapping["style_changes"].extend(polarity)

for _, v in mapping["technical"].items():
    for _, polarity in v.items():
        mapping["all_changes"].extend(polarity)
        mapping["technical_changes"].extend(polarity)

for _, v in mapping["composition"].items():
    for _, polarity in v.items():
        mapping["all_changes"].extend(polarity)
        mapping["composition_changes"].extend(polarity)


def filename2path(filename: str) -> str:
    threedirs = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:3]
    return "/".join(list(threedirs) + [filename])
