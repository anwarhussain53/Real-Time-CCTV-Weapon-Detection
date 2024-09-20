import sys
import os

# Add the directory of your app to the Python path
path = '/home/AnwarHussain/https://github.com/anwarhussain53/Real-Time-CCTV-Weapon-Detection'
if path not in sys.path:
    sys.path.append(path)

# Set the Flask app module
from app import app as application
