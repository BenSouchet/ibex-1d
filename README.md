# 1D IBEX : Image Barcode EXtractor

### Detect 1D Barcode(s) in Photographs, Extract & Straighten them!

A `Python 3` command line script to detect & extract barcode(s) in images, the script use **OpenCV** and **NumPy**.

<p align="center">
  <img src="https://user-images.githubusercontent.com/17025808/212469506-80761d45-934e-4c25-aeac-b591c0607fa3.png" width="600">
</p>

## Usage

Ensure you have a valid and working `python3`, `opencv` & `numpy` (Normally NumPy is a dependency of OpenCV, so you don't need to worry).

_Info : The script has been developed and tested on Python `3.11.0` & OpenCV `4.6.0`._

After cloning (or downloading) this repository:
```sh
python3 1d-ibex.py -i ~/Desktop/IMG_3212.png
```
The barcode images extracted will be saved into a newly created folder inside `./results/`, if nothing has been generated please check the log(s) in your terminal.

### Multiple images

You can pass one or more images/photographs to the script like this:
```sh
python3 1d-ibex.py -i ~/Desktop/IMG_3205.png ./object_12.jpg ~/Documents/photo_0345.jpeg
```
Inside the result subfolder, extracted paper sheets will be named `barcode_001.png`, `barcode_002.png`, `barcode_003.png`, ...

## Debug

You can visualize some steps of the sheet detection by adding the argument `-d` or `--debug` to the command:
```sh
python3 1d-ibex.py -i ~/Documents/product_03.jpeg -d
```
This will add debug images into the result subfolder.

## Errors / Warnings

In case of an error you should see a formated log message in your terminal telling you exactly what is the issue.
If the script crash or something don't work you can open an issue [here](https://github.com/BenSouchet/1d-ibex/issues).

## Improvements
- use `numpy.ascontiguousarray` on images arrays to use them as output or dest array, reducing allocations
- Adding more timing debug to detect bottle necks in the process
- Expose `is_document` and `max_quality` to script arguments

## Author / Maintainer

**1D IBEX** has been created and is currently maintained by [Ben Souchet](https://github.com/BenSouchet).

## Licenses

The code present in this repository is under [MIT license](https://github.com/BenSouchet/1d-ibex/blob/main/LICENSE).
