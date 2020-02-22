import subprocess
import tempfile

with tempfile.NamedTemporaryFile() as tmp:
    print("created temporary directory", tmp)
    # subprocess.call(["gegl", "-i", "/data/442284.jpg", "-o", tmp, ])
