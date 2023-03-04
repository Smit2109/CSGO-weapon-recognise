import numpy as np
import os
import csv
import sys
import matplotlib.pyplot as plt

from skimage import color
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from PIL import Image
from datetime import datetime


def add_text_to_csv(text):
    # Check if the file exists
    if len(text.shape) == 3:
        image_list = text.reshape(-1, text.shape[-1]).tolist()
    else:
        image_list = text.flatten().tolist()

    # Open a CSV file and write the image data
    with open('data.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(image_list)


def average_calculator(directory):
    folder = directory
    current = os.getcwd()
    folder = os.path.join(current, folder)

    images_list = []

    # parcurgem imaginile din folder
    for img_from_dir in os.listdir(folder):
        # imagine din folder
        image_folder = Image.open(os.path.join(folder, img_from_dir))
        image_grey = color.rgb2gray(image_folder)
        image_array2 = np.array(image_grey)
        #add_text_to_csv(image_array2)

        # adaugam noua imagine
        images_list.append(image_array2)

    # calculam media imaginilor din folder
    average_image = sum(images_list) / len(images_list)

    return average_image


def find_distance(image, average_image):
    image_array = np.array(image)

    pca = PCA(n_components=32)
    pca.fit(np.concatenate((image_array, average_image)))

    img1_pca = pca.transform(image_array)
    img2_pca = pca.transform(average_image)

    distance = pairwise_distances(img1_pca.reshape(1, -1), img2_pca.reshape(1, -1), metric='euclidean')

    distance = distance[0][0]

    return distance


average_AK = average_calculator("AK-47")
plt.imshow(average_AK, cmap='gray')
plt.colorbar()
plt.show()

average_AWP = average_calculator("AWP")
plt.imshow(average_AWP, cmap='gray')
plt.colorbar()
plt.show()

average_USP = average_calculator("USP")
plt.imshow(average_USP, cmap='gray')
plt.colorbar()
plt.show()

average_M4A4 = average_calculator("M4A4")
plt.imshow(average_M4A4, cmap='gray')
plt.colorbar()
plt.show()

average_weapon = (average_M4A4 + average_USP + average_AK + average_AWP) / 4
plt.imshow(average_weapon, cmap='gray')
plt.colorbar()
plt.show()


def try_test():
    folder_name = "Test"
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, folder_name)

    # parcurgem imaginile din folder
    for img in os.listdir(folder_path):
        start = datetime.now()
        image2 = Image.open(os.path.join(folder_path, img)).convert('L')
        image_array2 = np.array(image2)
        #add_text_to_csv(image_array2)

        minim = 100000

        var = find_distance(image_array2, average_AK)
        if minim > var:
            minim = var

            response = 'The test weapon is a AK'

        var = find_distance(image2, average_AWP)
        if minim > var:
            minim = var
            response = 'The test weapon is a AWP'

        var = find_distance(image2, average_USP)
        if minim > var:
            minim = var
            response = 'The test weapon is a USP'

        var = find_distance(image2, average_M4A4)
        if minim > var:
            # minim = var - nu mai este nevoie sa instantiem minim
            response = 'The test weapon is a M4A4'

        stop = datetime.now()
        print(str(stop-start))
        print(response)


try_test()

# incheiem scriptul cu return 0
sys.exit(0)
