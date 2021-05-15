import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("git+https://github.com/tensorflow/examples.git")
install("tensorflow")