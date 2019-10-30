# ClassifierAssistent
GUI application to help in batch sorting of image data for use with keras classifiers.

### File usage
Using TensorFlow backend.
Usage: batch_classify.py [OPTIONS]

  This script reads a pre-trained keras model and tries to use good guesses
  from this model to speed up subsequent manual classification. It parses
  all images in a given folder and groups these into similar classes. A
  batch of images (default 6x5 grid) which are tagged as the same class is
  presented to the user. By pressing any key (except Q) these are accepted
  as the given class and moved to the desired location. By selecting one or
  more images with the mouse, you can reject these suggestions by the
  classifier.

  Pressing Q will terminate the program.

Options:
  --batchsize INTEGER...     Number of simultanuous images to classify (grid
                             size: 2 numbers)
  --modelname TEXT           Filename of keras classifier model
  --verbose BOOLEAN          Verbose output. Print additional debug
                             information
  --unsorted_directory TEXT  Files will be bulk classified and MOVED from here
  --sorted_directory TEXT    Desitation folder for sorted dataset
  --help                     Show this message and exit.

### Dependencies
The script relies on the following list of python modules, which are all available through pip

* numpy
* keras
* tensorflow
* opencv
* click
