import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def display_samples(samples, labels):
    plt.figure(figsize=(10,10))

    for i in range(25):
        idx = np.random.randint(0,samples.shape[0])
        img = samples[idx]
        label = labels[idx]

        plt.subplot(5,5, i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')

    plt.show()

def split_data(path_to_data, train_path, val_path, split_size=0.1):
    
    folders = os.listdir(path_to_data)

    for folder in folders:
        
        full_path = os.path.join(path_to_data, folder)
        image_paths = glob.glob(os.path.join(full_path, '*.png'))

        x_train, x_val = train_test_split(image_paths, test_size=split_size)

        for x in x_train:

            path_to_folder = os.path.join(train_path, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)

        for x in x_val:

            path_to_folder = os.path.join(val_path, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)


def order_test_set(img_path, csv_path):
    
    testset = {}

    try:
        with open(csv_path, 'r') as csvfile:
            
            reader = csv.reader(csvfile, delimiter=',')

            for i, row in enumerate(reader):
                if i == 0:
                    continue

                img_name = row[-1].replace('Test/', '')
                label = row[-2]

                path_to_folder = os.path.join(img_path, label)

                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                img_full_path = os.path.join(img_path, img_name)
                shutil.move(img_full_path, path_to_folder)

    except:
        print('[INFO]: Error reading csv file.')


    print(testset)


def create_generators(batch_size, train_path, val_path, test_path):
    train_preprocessor = ImageDataGenerator(
        rescale=1/255. ,
        rotation_range=10,
        width_shift_range=0.1
    )

    test_preprocessor = ImageDataGenerator(
        rescale=1/255.
    )

    train_generator = train_preprocessor.flow_from_directory(
        train_path,
        class_mode='categorical',
        target_size=(60, 60),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = test_preprocessor.flow_from_directory(
        val_path,
        class_mode='categorical',
        target_size=(60, 60),
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size
    )

    test_generator = test_preprocessor.flow_from_directory(
        test_path,
        class_mode='categorical',
        target_size=(60, 60),
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size
    )

    return train_generator, val_generator, test_generator
