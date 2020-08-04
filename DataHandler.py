import glob
import shutil
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np



food_classes = ['bread', 'dairy_product', 'dessert', 'egg', 'fried_food', 'meat', 
                'noodles_pasta', 'rice', 'seafood', 'soup']


def split_data_into_class_folders(path_to_data, class_id):

    imgs_paths = glob.glob(path_to_data + '*.jpg')
    for path in imgs_paths:
        basename = os.path.basename(path)
        class_nbr = int(basename[0])

        if class_nbr == class_id:
            path_to_save = os.path.join(path_to_data, food_classes[class_id])

            if not os.path.isdir(path_to_save):
                os.makedirs(path_to_save)

            shutil.move(path, path_to_save)


def visualise_some_images(path_to_data):

    #onlyfiles = [f for f in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, f))]
    imgs_paths = []
    labels = []

    for r, d, f in os.walk(path_to_data):
        for file in f:
            if file.endswith(".jpg"):
                imgs_paths.append(os.path.join(r, file))
                labels.append(os.path.basename(r))

    fig = plt.figure()
    

    for i in range(16):
        chosen_index = random.randint(0, len(imgs_paths)-1)
        chosen_img = imgs_paths[chosen_index]
        chosen_label = labels[chosen_index]

        ax1 = fig.add_subplot(4, 4, i+1)
        ax1.title.set_text(chosen_label)
        ax1.imshow(Image.open(chosen_img))

    fig.tight_layout(pad=0.1)    
    plt.show()


def get_images_sizes(path_to_data):

    imgs_paths = []
    counter = 0
    widths = []
    heights = []

    for r, d, f in os.walk(path_to_data):
        for file in f:
            if file.endswith(".jpg"):
                counter += 1
                img = Image.open(os.path.join(r, file))
                widths.append(img.size[0])
                heights.append(img.size[1])
                img.close()

    mean_width = sum(widths) / len(widths)
    mean_height = sum(heights) / len(heights)
    median_width = np.median(widths)
    median_height = np.median(heights)

    return mean_width, mean_height, median_width, median_height




if __name__ == '__main__':

    split_data_switcher = False
    visualize_data_switcher = False

    path_to_train_data = '/home/nourislam/Downloads/218640_473358_bundle_archive/food-11/training/'
    path_to_val_data  = '/home/nourislam/Downloads/218640_473358_bundle_archive/food-11/validation/'
    path_to_eval_data  = '/home/nourislam/Downloads/218640_473358_bundle_archive/food-11/evaluation/'

    if split_data_switcher:

        for i in range(11):
            split_data_into_class_folders(path_to_train_data, i)
        for i in range(11):
                split_data_into_class_folders(path_to_val_data, i)
        for i in range(11):
                split_data_into_class_folders(path_to_eval_data, i)


    if visualize_data_switcher:
        visualise_some_images(path_to_train_data)

    mean_width, mean_height, median_width, median_height = get_images_sizes(path_to_train_data)
    print(f"mean_width = {mean_width}")
    print(f"mean_height = {mean_height}")
    print(f"median_width = {median_width}")
    print(f"median_height = {median_height}")
    
    