# Copyright (C) 2019 SINTEF Digital,
# Department of Mathematics and Cybernetics
#
# Contact information: Kjetil Andre Johannessen
# E-mail: Kjetil.Johannessen@sintef.no
#
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Public License v3 for more details.
#
# You should have received a copy of the GNU Public Licence v3
# License along with the script. If not, see
# <http://www.gnu.org/licenses/>.
#

import cv2
import numpy as np
import sys
import click
import os
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator

@click.command()
@click.option('--batchsize', nargs=2, type=int, default=(6,5),
              help='Number of simultanuous images to classify (grid size: 2 numbers)')
@click.option('--modelname', type=str, default='model.h5',
              help='Filename of keras classifier model')
@click.option('--verbose', type=bool, default=False,
              help='Verbose output. Print additional debug information')
@click.option('--unsorted_directory', prompt='Unsorted dataset directory',
              help='Files will be bulk classified and MOVED from here')
@click.option('--sorted_directory', prompt='Target directory',
              help='Desitation folder for sorted dataset')
def run(batchsize, modelname, unsorted_directory, sorted_directory, verbose):
    """ This script reads a pre-trained keras model and tries to use good guesses from this model
        to speed up subsequent manual classification. It parses all images in a given folder and
        groups these into similar classes. A batch of images (default 6x5 grid) which are tagged as
        the same class is presented to the user. By pressing any key (except Q) these are accepted
        as the given class and moved to the desired location. By selecting one or more images with
        the mouse, you can reject these suggestions by the classifier.

        Pressing Q will terminate the program.
    """

    # load computational model
    classifier = load_model(modelname)

    # Disiplay running information
    print(f'Batch Size: {batchsize}')
    print(f'Model Name: {modelname}')
    img_size = classifier.get_input_shape_at(0)[1:3]
    numb_classes = classifier.get_output_shape_at(-1)[-1]
    print(f'Image size: {img_size}')
    print(f'Number of classes: {numb_classes}')
    print(f'Unsorted directory: {unsorted_directory}')
    print(f'Sorted directory: {sorted_directory}')

    # print model parameters
    if verbose:
        print(classifier.summary())

    if not os.path.exists(sorted_directory):
        os.makedirs(sorted_directory)


    # Function to find (i,j) index of mini-image clicked and slightly shrink it to
    # show it as selected. Also creates a white border around these images
    # Results after function call is that the boolean grid 'selected' is updated
    def mouse_click(event, x, y, flags, param):
        border = 7
        i = y//img_size[0]
        j = x//img_size[1]
        imsiz = np.array(img_size) # here for more convenient arithmetic (arrays better than tuples)
        if event == cv2.EVENT_LBUTTONDOWN:
            print(i,j)
            width  = slice(i*img_size[0],(i+1)*img_size[0], None)
            height = slice(j*img_size[1],(j+1)*img_size[1], None)
            width_s  = slice(i*img_size[0]+border,(i+1)*img_size[0]-border, None)
            height_s = slice(j*img_size[1]+border,(j+1)*img_size[1]-border, None)

            I[width,height] = 1
            small_img  = np.zeros(imsiz-2*border)
            large_img = img[i*batchsize[0]+j]
            small_img = cv2.resize(large_img, small_img.T.shape)
            selected[i,j] = not selected[i,j]

            if selected[i,j]:
                I[width_s, height_s] = small_img
            else:
                I[width, height]     = large_img
            cv2.imshow(f'CLASS {lookfor}', I)

    # helper function to move all accepted (unselected) files to new location
    def move_files():
        k = 0
        if verbose:
            print(f'Class {lookfor}')
        if not os.path.exists(f'{sorted_directory}\{lookfor}'):
            os.makedirs(f'{sorted_directory}\{lookfor}')

        for i in range(batchsize[1]):
            for j in range(batchsize[0]):
                if not selected[i][j]:
                    basename = os.path.basename(filename[k])
                    if verbose:
                        print(f'  {filename[k]}')
                    os.rename(f'{unsorted_directory}\{filename[k]}', f'{sorted_directory}\{lookfor}\{basename}')
                k += 1



    # initialize all variables, some of these are used in the two functions above
    test_datagen  = ImageDataGenerator(rescale=1/256)
    n = 2*np.prod(batchsize)
    img_stream  = test_datagen.flow_from_directory(unsorted_directory,  target_size=img_size, batch_size=n, class_mode='categorical', color_mode='grayscale', shuffle=False)
    category = dict((v,k) for k,v in img_stream.class_indices.items())

    filename = []
    img = []
    idx = 0
    lookfor = None
    files_classified = 0
    I = np.zeros((img_size[0]*batchsize[1], img_size[1]*batchsize[0]))

    # Iterate over all input images
    try:
        for (i,x) in enumerate(img_stream):
            images, classes = x
            percentage_predictions = classifier.predict(images)
            int_prediction = np.argmax(percentage_predictions, axis=1)
            # Find most frequent prediction
            if lookfor is None:
                lookfor = np.argmax(np.bincount(int_prediction))

            # Assemble a grid of images which all have the same class prediction
            for j in range(int_prediction.shape[0]):
                if int_prediction[j]==lookfor:
                    # Exit condition: last image parsed
                    if len(img_stream.filenames) <= i*n+j:
                        total_files = len(img_stream.filenames)
                        print(f'Files moved: {files_classified} out of a total of {total_files}')
                        return
                    filename.append(img_stream.filenames[i*n + j])
                    i0 = slice((idx// batchsize[0])*img_size[0],(idx// batchsize[0]+1)*img_size[0], None)
                    j0 = slice((idx % batchsize[0])*img_size[1],(idx % batchsize[0]+1)*img_size[1], None)
                    I[i0,j0] = images[j,:,:,0]
                    img.append(images[j,:,:,0])
                    idx += 1

                # When enough images are found, show these to the user
                if idx == np.prod(batchsize):
                    selected = np.zeros(batchsize[::-1], dtype=bool)

                    cv2.imshow(f'CLASS {lookfor}', I)
                    cv2.setMouseCallback(f'CLASS {lookfor}', mouse_click)

                    # Exit condition: User pressed the Q-key
                    key = cv2.waitKey(0) & 0xFF
                    if key==ord('q'):
                        cv2.destroyAllWindows()
                        total_files = len(img_stream.filenames)
                        print(f'Files moved: {files_classified} out of a total of {total_files}')
                        return

                    # Move files to correct folder, reset variables and start looking up a new batch
                    # of similar predictions
                    move_files()
                    files_classified += np.prod(batchsize) - np.sum(selected)
                    filename = []
                    img = []
                    idx = 0
                    lookfor = None
                    cv2.destroyAllWindows()

    # Exit condition: Not really sure why this triggers, but it seem to be related to the end condition
    #                 of the image stream generator (when the system moves files out of the generators folder)
    except FileNotFoundError:
        total_files = len(img_stream.filenames)
        print(f'Files moved: {files_classified} out of a total of {total_files}')



# Run this script from the commandline
if __name__ == '__main__':
    run()
