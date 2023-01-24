# IBEX 1D : Image 1D Barcode EXtractor

### Detect 1D Barcode(s) in Photographs, Extract & Straighten them!

A `Python 3` command line script to detect & extract barcode(s) in images, using **OpenCV** and **NumPy**.

<p align="center">
  <img alt="Example of barcode extraction" src="https://user-images.githubusercontent.com/17025808/212469506-80761d45-934e-4c25-aeac-b591c0607fa3.png" width="600">
</p>

## Install

### Via PIP

```sh
pip3 install ibex_1d
```

Then (if you have for example a photograph located at `~/Desktop/IMG_3212.png`):

```sh
ibex_1d -i ~/Desktop/IMG_3212.png
```

### Via Source Code
Download the [latest release](https://github.com/BenSouchet/ibex_1d/releases) or clone/download this repository, then:
```sh
python3 ibex_1d.py -i ~/Desktop/IMG_3212.png
```
**NOTE**: Using this method you will need to call the script with `python3 ibex_1d.py ...`

## Results

The barcode images extracted will be saved into a newly created folder inside `./results/`, if nothing has been generated please check the log(s) in your terminal.

## Multiple images

You can pass one or more images/photographs to the script like this:
```sh
ibex_1d -i ~/Desktop/IMG_3205.png ./object_12.jpg ~/Documents/photo_0345.jpeg
```
Inside the result sub-folder, extracted paper sheets will be named `barcode_001.png`, `barcode_002.png`, `barcode_003.png`, ...

## Incorrect result ?
If the barcode hasn't been extracted (or the resulting barcode image isn't good) this can be due to the OTSU threshold method.
You can try using the script with the argument `-a` (or `--adaptive-threshold`):
```sh
ibex_1d -i ~/Desktop/IMG_3205.png -a
```
This threshold method isn't set as default because it's slower than OTSU.

## Debug

You can visualize some steps of the sheet detection by adding the argument `-d` or `--debug` to the command:
```sh
ibex_1d -i ~/Documents/product_03.jpeg -d
```
This will add debug images into the result sub-folder.

## Errors / Warnings

In case of an error you should see a formatted log message in your terminal telling you exactly what is the issue.
If the script crash or something don't work you can open an issue [here](https://github.com/BenSouchet/ibex_1d/issues).


## Author / Maintainer

**IBEX 1D** has been created and is currently maintained by [Ben Souchet](https://github.com/BenSouchet).

## Licenses

The code present in this repository is under [MIT license](https://github.com/BenSouchet/ibex_1d/blob/main/LICENSE).
