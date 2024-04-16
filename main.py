import tensorflow as tf
import numpy as np
import os
import keras
from keras import models, layers, losses, callbacks, utils, optimizers
from keras.applications import ResNet50V2
import matplotlib.pyplot as plt

save_format = 'keras'

if not tf.config.list_physical_devices('GPU') == []:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

us_classes = {
    0:'Stop',
    1:'Yield',
    2:'Stop here for pedestrians',
    3:'Yield here for pedestrians',
    4:'Speed limit',
    5:'No right turn',
    6:'No left turn',
    7:'No U-turn',
    8:'No left or U-turn',
    9:'No straight through',
    10:'Straight ahead only',
    11:'Option sign for left turn or Straight',
    12:'Option sign for right turn or Straight',
    13:'Right turn only',
    14:'Left turn only',
    15:'Right lane must turn right',
    16:'Left lane must turn left',
    17:'Pedestrian crossing',
    18:'Do not pass',
    19:'Begin right turn lane yield to bikes',
    20:'Keep right',
    21:'Keep left',
    22:'Slippery when wet',
    23:'Wrong way',
    24:'One way',
    25:'Road closed',
    26:'Railroad crossing',
    27:'Railroad',
    28:'School',
    29:'School speed zone ahead',
    30:'End school zone',
    31:'Sharp turn to the left',
    32:'Sharp turn to the right',
    33:'Curve to the left',
    34:'Curve to the right',
    35:'Hairpin curve to the left',
    36:'Hairpin curve to the right',
    37:'Right lane ends',
    38:'Merging lanes',
    39:'Crossroads',
    40:'Side road at a perpendicular angle to the left',
    41:'Side road at a perpendicular angle to the right',
    42:'T-shaped junction',
    43:'Left lane ends',
    44:'Right lane ends',
    45:'Divided highway begins',
    46:'Divided highway ends',
    47:'Two-way traffic',
    48:'Bump',
    49:'Dip'
}

dataset_use = 'us'
if dataset_use == 'us':
    num_classes = 50
    data_path = './us_data'
    classes = us_classes
else:
    num_classes = 43
    data_path = './data'
    classes = de_classes

if not (os.path.isfile(f'./trained_model.{save_format}')):
    print("No saved model found. Training...")

    class_names = list(range(num_classes))
    for i in range(len(class_names)):
        class_names[i] = str(class_names[i])
    
    train_dataset = utils.image_dataset_from_directory(f"{data_path}/Train", class_names=class_names, image_size=(224,224), subset="training", validation_split=0.2, seed=42)
    val_dataset = utils.image_dataset_from_directory(f"{data_path}/Train", class_names=class_names, image_size=(224,224), subset="validation", validation_split=0.2, seed=42)

    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=3)

    conv_base = ResNet50V2(include_top=False, input_shape=(224,224,3), classes=num_classes, classifier_activation="softmax")
    conv_base.summary()

    inputs = layers.Input(shape=(224,224,3))
    mod = layers.Rescaling(1./255)(inputs)
    mod = conv_base(inputs)
    mod = layers.GlobalAveragePooling2D()(mod)
    mod = layers.Dense(256, "relu")(mod)
    mod = layers.Dropout(0.5)(mod)
    outputs = layers.Dense(num_classes, "softmax")(mod)

    model = models.Model(inputs, outputs)

    model.summary()

    opt = optimizers.Adam(0.00001)
    model.compile(optimizer=opt, 
                loss=losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    model.fit(train_dataset, epochs=5, batch_size=32, callbacks=[early_stop], validation_data=val_dataset)

    acc = np.array(model.history.history['accuracy'])
    val_acc = np.array(model.history.history['val_accuracy'])
    loss = np.array(model.history.history['loss'])
    val_loss = np.array(model.history.history['val_loss'])

    model.save(f'./trained_model.{save_format}')

    np.save("history/acc.npy", acc)
    np.save("history/val_acc.npy", val_acc)
    np.save("history/loss.npy", loss)
    np.save("history/val_loss.npy", val_loss)
    print("Model loaded")