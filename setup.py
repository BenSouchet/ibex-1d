"""Setup file for 1D IBEX"""

# Always prefer setuptools over distutils
from setuptools import setup
import pathlib
from ibex_1d import VERSION

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="ibex_1d",
    version=VERSION,
    description="Image 1D Barcode EXtractor - Detect and Extract 1D Barcode(s) in photographs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BenSouchet/ibex_1d",
    author="Ben Souchet",
    author_email="contact@bensouchet.dev",
    license='MIT',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="opencv, computer-vision, barcode, detection, extraction, roi, image-recognition",
    py_modules=["ibex_1d"],
    python_requires=">=3.7, <4",
    install_requires=["numpy", "opencv-python"],
    entry_points={
        "console_scripts": [
            "ibex_1d=ibex_1d:main",
        ],
    },
    project_urls={
        "Issue Reports": "https://github.com/BenSouchet/ibex_1d/issues",
        "Source": "https://github.com/BenSouchet/ibex_1d",
    },
)
