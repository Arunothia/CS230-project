import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def upgrade(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])


install("git+https://github.com/tensorflow/examples.git")
install("matplotlib")
install("tkinter")
install("IPython")
install("tensorflow")
upgrade("tensorflow")
install("tensorflow-datasets")