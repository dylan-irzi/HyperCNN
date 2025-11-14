# setup.py
from setuptools import setup, find_packages

# Lee el contenido del README.md para la descripción larga
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Lee los requisitos del requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hypercnn", # El nombre que usarás para instalar con pip install hypercnn
    version="0.1.0", # Cambia esta versión según vayas actualizando
    author="Juan Castro", # Tu nombre o el del autor principal
    author_email="dylan.irzi1@gmail.com", # Tu email
    description="A convolutional neural network with hyperbolic topology for efficient visual recognition.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dylan-irzi/HyperCNN", # URL del repositorio
    packages=find_packages(), # Encuentra automáticamente los paquetes (tu carpeta hypercnn/)
    classifiers=[
        "Development Status :: 3 - Alpha", # O 4 - Beta, 5 - Production/Stable
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License", # Asegúrate de que coincida con tu LICENSE
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8', # Versión mínima de Python requerida
    install_requires=requirements, # Dependencias del requirements.txt
)
