from setuptools import setup, find_packages

setup(
    name="simtools",
    version="0.1.0",
    packages=find_packages(),
    author="Tom Groß",
    install_requires=[
        "numpy",
        "pyvisgen",
        "radiotools",
        "casatools",
        "casatasks",
        "casadata",
        "python-casacore",
        "astropy",
        "matplotlib",
        "casadata",
    ],
)
