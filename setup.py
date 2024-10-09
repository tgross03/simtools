from setuptools import setup, find_packages

setup(
    name="simtools",
    version="0.1.0",
    packages=find_packages(),
    author="Tom Groß",
    install_requires=[
        "numpy",
        "matplotlib",
        "pyvisgen",
        "radiotools",
        "radio_stats",
        "casatools",
        "casatasks",
        "casadata",
        "astropy",
    ],
)
