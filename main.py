import tensorflow as tf
import numpy as np
import os
from keras import models, layers, losses, callbacks, utils, backend
from keras.applications import ResNet50V2
import matplotlib.pyplot as plt
import process_data
from sklearn.utils import shuffle

save_format = 'keras'

if not tf.config.list_physical_devices('GPU') == []:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

def interpret_results(predict_results: np.ndarray):
    for index, item in enumerate(predict_results):
        max_proba = 0
        max_class = -1
        for classid, class_proba in enumerate(item):
            if class_proba > max_proba:
                max_proba = class_proba
                max_class = classid
        print(f"Predicted output for input {index}: {classes[max_class]}({max_class}), {max_proba:.3f}")

class DataSequence(utils.Sequence):
    def __init__(self, batch_size, x_dat, y_dat):
        self.batch_size = batch_size
        self.x = x_dat
        self.y = y_dat

    def __len__(self):
        return int(np.floor(len(self.x)/self.batch_size))
    
    def __getitem__(self, index):
        low = index*self.batch_size
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]

        return batch_x, batch_y

if not (os.path.isfile(f'./trained_model.{save_format}')):
    print("No saved model found. Training...")

    pre_data_path = './data/Preprocessed/'
    if not (os.path.isfile(f'{pre_data_path}x_data.npy') and os.path.isfile(f'{pre_data_path}y_data.npy')):
        process_data.preprocess_data()

    x_data = np.load(f'{pre_data_path}x_data.npy')
    y_data = np.load(f'{pre_data_path}y_data.npy')

    print(x_data.shape)
    print(y_data.shape)

    X, y = shuffle(x_data, y_data)

    early_stop = callbacks.EarlyStopping(monitor="loss", patience=3)

    data_sequence = DataSequence(process_data.BATCH_SIZE, X, y)

    conv_base = ResNet50V2(include_top=False, input_shape=(224,224,3), classes=43, classifier_activation="softmax")
    conv_base.summary()

    inputs = layers.Input(shape=(224,224,3))
    mod = conv_base(inputs)
    mod = layers.GlobalAveragePooling2D()(mod)
    mod = layers.Dense(256, "relu")(mod)
    mod = layers.Dropout(0.5)(mod)
    outputs = layers.Dense(43, "softmax")(mod)

    model = models.Model(inputs, outputs)

    # model = models.Sequential()
    # model.add(conv_base)
    # model.add(layers.GlobalAveragePooling2D())
    # model.add(layers.Dense(units=256, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(units=43, activation='softmax'))
    # model.build()

    print(model.layers[2].input_spec)

    model.summary()

    # AlexNet
    # model = models.Sequential()
    # model.add(layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(224, 224, 3)))
    # model.add(layers.MaxPooling2D((3, 3), strides=2))
    # model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
    # model.add(layers.MaxPooling2D((3, 3), strides=2))
    # model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Flatten())
    # model.add(layers.Dense(4096, activation='relu'))
    # model.add(layers.Dense(1024, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(43, activation='softmax'))
    # model.summary()

    model.compile(optimizer="adam", 
                loss=losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

    model.fit(data_sequence, epochs=1, batch_size=process_data.BATCH_SIZE, callbacks=[early_stop])

    acc = model.history.history['accuracy']
    print(acc)

    model.save(f'./trained_model.{save_format}')
else:
    print("Loading Saved Model from File")
    model = models.load_model(f'./trained_model.{save_format}')
print("Model finished training/loading")

while(True):
    img = input()
    if(img == ""):
        break
    split = img.split(',')
    try:
        classid = split[0]
        num = split[1]
    except IndexError:
        continue 
    try:
        filename = os.listdir(f'data/Train/{classid}')[int(num)]
    except (ValueError, IndexError):
        continue
    print(filename)
    single_img = process_data.load_single_image(f'data/Train/{classid}/{filename}')
    result = model.predict(tf.expand_dims(single_img, axis=0))
    interpret_results(result)