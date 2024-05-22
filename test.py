from keras import models, utils, layers
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

test_path = "us_data/Test"

class DataSequence(utils.Sequence):
    def __init__(self, batch_size, path):
        self.batch_size = batch_size
        self.x = os.listdir(path)

    def __len__(self):
        return int(np.floor(len(self.x)/self.batch_size))
    
    def __getitem__(self, index):
        low = index*self.batch_size
        high = min(low + self.batch_size, len(self.x))
        
        batch_x = []
        batch_y = []

        for i in range(low, high):
            fname = self.x[i]
            img = utils.img_to_array(utils.load_img(f"{test_path}/{fname}"))
            img = tf.image.resize_with_pad(img, 224, 224)
            batch_x.append(img)
            batch_y.append(int(fname.split("_")[0]))

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y

def main():
    if not os.path.isfile("trained_model.keras"):
        return
    if not os.path.isdir(test_path):
        return
    if len(os.listdir(test_path)) == 0:
        return
    model = models.load_model("trained_model.keras")
    print("Model loaded")

    num_images = len(os.listdir(test_path))
    sequence = DataSequence(32, test_path)

    # wrong_x = []
    # wrong_y_guess = []
    # wrong_y_real = []
    # wrong_count = 0

    wrong_counts = np.zeros(50, np.int32)
    total_per_class = np.zeros(50, np.int32)

    for x_batch, y_batch in sequence:
        predictions = model.predict(x_batch, batch_size=32)
        for i in range(len(x_batch)):
            predicted_class = np.argmax(predictions[i], axis=-1)
            real_class = y_batch[i]
            total_per_class[real_class] += 1
            if(not predicted_class == real_class):
                wrong_counts[real_class] += 1 
        # for i in range(20):
        #     predicted_class = np.argmax(predictions[i], axis=-1)
        #     real_class = y_batch[i]
        #     if (not predicted_class == real_class):
        #         wrong_x.append(x_batch[i])
        #         wrong_y_guess.append(predicted_class)
        #         wrong_y_real.append(y_batch[i])
        #         wrong_count += 1
    
    wrong_count = 0
    total_count = 0
    for i in range(50):
        acc = 1 - (wrong_counts[i]/max(total_per_class[i], 1))
        print(f"{i},{acc}")
        wrong_count += wrong_counts[i]
        total_count += total_per_class[i]
    print(f"Total accuracy: {1 - wrong_count/total_count}")
    
    # print(f"Wrong guesses: {wrong_count} / Count: {len(sequence)*32}")
    # print(1 - wrong_count/(len(sequence)*32))

    # fig = plt.figure(figsize=(8,8))
    # for i in range(wrong_count):
    #     plt.subplot(4,5,i%20+1)
    #     plt.imshow(wrong_x[i]/255)
    #     plt.title(f"Guessed: {wrong_y_guess[i]}\nActual: {wrong_y_real[i]}")
    #     plt.axis("off")
    #     if (i+1)%20 == 0 or i == wrong_count - 1:
    #         fig.tight_layout()
    #         plt.show()
    #         fig = plt.figure(figsize=(8,8))

if __name__ == "__main__":
    main()