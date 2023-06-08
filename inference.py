"""# Classify images from input folder"""

# https://stackoverflow.com/questions/43017017/keras-model-predict-for-a-single-image
import os, re
# You can disable all debugging logs using os.environ -> removing tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
import tensorflow as tf
#print(tf.__version__)
from keras import layers, models, datasets
from keras.models import load_model
import argparse
import cv2
import numpy as np
from typing import List


LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'A', 'B', 'D', 'E', 'F', 'G', 'H', 'N', 'Q', 'R', 'T'] #Modify here to get only uppercase
#          'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']


# def extract_numbers(filename):
#     return int(re.search(r'\d+', filename).group())


def sort_filenames(folder_path) -> List[str]:
    filenames = os.listdir(folder_path)
    # sorted_filenames = sorted(filenames, key=extract_numbers)
    sorted_filenames = sorted(filenames, key=str.lower)
    # print(f"Folder {folder_path} contains [{len(sorted_filenames)}] files:\n{sorted_filenames}")
    return sorted_filenames


def get_filename_ext(filepath):
    directory, file_name = os.path.split(filepath)
    file_name, extension = os.path.splitext(file_name)
    return file_name, extension


def load_image(image_path, target_shape):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
    image = cv2.resize(image, target_shape[:2])

    # Add extra dimension for channel (1)
    image_data = np.expand_dims(image, axis=-1)

    return image_data


def process_images(folder_path, target_shape=(28, 28, 1)):
    images = []
    image_paths = []
    filenames = sort_filenames(folder_path)
    for filename in filenames:
        filename_wo_ext, ext = get_filename_ext(f"{folder_path}/{filename}")
        if ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]:
            image_paths.append(os.path.join(folder_path, filename))
            image_data = load_image(os.path.join(folder_path, filename), target_shape)
            image_data = np.expand_dims(image_data, axis=0)
            images.append(image_data)

    result = np.array(images)
    result.reshape(-1, 28, 28, 1)

    return result, image_paths


def preprocess_input(input_path):
    folder_path = input_path
    images, paths = process_images(folder_path)

    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = "model.h5"

    model = load_model(f"{model_dir}/{model_name}", compile=False)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )

    # https://stackoverflow.com/questions/42666046/loading-a-trained-keras-model-and-continue-training
    for img, p in zip(images, paths):
        result = LABELS[int(np.argmax(model.predict(img, verbose=0), axis=1))]
        # print(f"{result} {ord(result)}, {p}")
        print(f"{ord(result)}, {p}") ## [character ASCII index in decimal format], [POSIX path to image sample]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input Preprocessing Script")
    parser.add_argument("--input", help="Path to the input data")

    args = parser.parse_args()

    if args.input:
        preprocess_input(args.input)
    else:
        print("No input path provided. Please provide the --input argument.")