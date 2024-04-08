import os
import numpy as np
import keras
import matplotlib.pyplot as plt

data_dir = './us_data'
train_path = f'{data_dir}/Train/'
test_path = f'{data_dir}/Test/'
preprocessed_data_path = f'{data_dir}/Preprocessed/'
if not os.path.exists(preprocessed_data_path):
    os.mkdir(preprocessed_data_path)

NUM_CATEGORIES = len(os.listdir(train_path))

IMG_HEIGHT = 224
IMG_WIDTH = 224
channels = 3

BAGGING_COUNT = 250
BATCH_SIZE = 32

def preprocess_data(debug=False):
    if debug:
        plot = plt.figure()

    x_data = []
    y_data = []
    for classid in os.listdir(train_path):
        i = 0
        print(f"Files in class {classid}: {len(os.listdir(train_path + classid))}")
        for filename in os.listdir(train_path + classid):
            img = keras.utils.load_img(f"{train_path+classid}/{filename}", 
                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                    interpolation="lanczos")
            img_arr = keras.utils.img_to_array(img=img)/255
            x_data.append(img_arr)
            class_int = int(classid)
            y_instance = class_int
            y_data.append(y_instance)
            if debug:
                if i == 0:
                    plt.subplot(5, 10, class_int + 1)
                    plt.axis('off')
                    plt.imshow(img_arr)
            i += 1
            if i > BAGGING_COUNT - 1:
                break
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    if debug:
        plt.show()

    np.save(preprocessed_data_path + "x_data", x_data)
    np.save(preprocessed_data_path + "y_data", y_data)

def load_single_image(path: str):
    img = keras.utils.load_img(path,
                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                               interpolation="bicubic")
    img_arr = keras.utils.img_to_array(img=img)/255
    return np.array(img_arr)

def main():
    print("Debugging Output")
    preprocess_data(True)

if __name__ == "__main__":
    main()