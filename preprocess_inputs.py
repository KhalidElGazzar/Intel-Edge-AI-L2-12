import cv2
import numpy as np

def pose_estimation(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related pose estimation model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the pose estimation model
    '''
    Inputs
    name: "input" , shape: [1x3x256x456] - An input image in the format [BxCxHxW], where:
    B - batch size
    C - number of channels
    H - image height
    W - image width. Expected color order is BGR.
    '''

    height = 256 
    width = 456

    preprocessed_image = cv2.resize(input_image, (width, height))
    preprocessed_image = preprocessed_image.transpose((2,0,1))
    preprocessed_image = preprocessed_image.reshape(1, 3, height, width)

    return preprocessed_image


def text_detection(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related text detection model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the text detection model

    """
    Inputs
    name: "input" , shape: [1x3x768x1280] - An input image in the format [BxCxHxW], where:

    B - batch size
    C - number of channels
    H - image height
    W - image width
    Expected color order - BGR.
    """

    height = 768
    width = 1280

    preprocessed_image = cv2.resize(input_image, (width, height))
    preprocessed_image = preprocessed_image.transpose((2,0,1))
    preprocessed_image = preprocessed_image.reshape(1, 3, height, width)

    return preprocessed_image


def car_meta(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related car metadata model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the car metadata model
    

    '''
    Inputs
    name: "input" , shape: [1x3x72x72] - An input image in following format [1xCxHxW], where:
    - C - number of channels
    - H - image height
    - W - image width.
    Expected color order - BGR.

    Outputs
    name: "color", shape: [1, 7, 1, 1] - Softmax output across seven color classes [white, gray, yellow, red, green, blue, black]
    name: "type", shape: [1, 4, 1, 1] - Softmax output across four type classes [car, bus, truck, van]
    '''


    height = 72
    width = 72

    preprocessed_image = cv2.resize(input_image, (width, height))
    preprocessed_image = preprocessed_image.transpose((2,0,1))
    preprocessed_image = preprocessed_image.reshape(1, 3, height, width)


    return preprocessed_image
