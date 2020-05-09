from pathlib import Path
import yaml

yaml_outdir = Path("/home") / "stud" / "pfister" / "eclipse-workspace" / "NIAA" / "k8s" / "prepare_images"
template_file = yaml_outdir / "template.yaml"
pexels_dir = Path("/scratch") / "stud" / "pfister" / "NIAA" / "pexels"
img_dir = pexels_dir / "images"
out_dir = pexels_dir / "edited_images"
xmp_dir = pexels_dir / "xmps"

pexels_docker_dir = Path("/scratch") / "pexels"
img_docker_dir = pexels_docker_dir / "images"
out_docker_dir = pexels_docker_dir / "edited_images"
xmp_docker_dir = pexels_docker_dir / "xmps"

with open(template_file) as template:
    yaml_file = yaml.load(template)

for subfolder in xmp_dir.iterdir():
    parameter = subfolder.name
    (yaml_outdir / parameter).mkdir(exist_ok=True)
    for xmp_file in subfolder.iterdir():
        change = xmp_file.stem
        (out_dir / parameter / change).mkdir(parents=True, exist_ok=True)

        job_name = f"prepare-{parameter}-{'m-' if float(change) < 0 else ''}{change.replace('.', '').replace('-', '')}"
        yaml_file["metadata"]["name"] = job_name
        yaml_file["spec"]["template"]["spec"]["containers"][0]["name"] = job_name

        yaml_file["spec"]["template"]["spec"]["containers"][0]["command"][1] = str(img_docker_dir)
        yaml_file["spec"]["template"]["spec"]["containers"][0]["command"][2] = str(xmp_docker_dir / parameter / xmp_file.name)
        yaml_file["spec"]["template"]["spec"]["containers"][0]["command"][3] = str(out_docker_dir / parameter / xmp_file.stem / "$(FILE_NAME).jpg")

        with open(yaml_outdir / parameter / f"{xmp_file.stem}.yaml", "w") as out:
            yaml.dump(yaml_file, out, sort_keys=False)
