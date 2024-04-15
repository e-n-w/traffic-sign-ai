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

de_classes = {
    0:'Speed limit (20km/h)',
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
    42:'End no passing veh > 3.5 tons'
}

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

def interpret_result(predict_results: np.ndarray):
    for index, item in enumerate(predict_results):
        max_proba = 0
        max_class = -1
        for classid, class_proba in enumerate(item):
            if class_proba > max_proba:
                max_proba = class_proba
                max_class = classid
        return (max_class, max_proba, index)

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
    
def load_single_image(path: str):
    img = keras.utils.load_img(path,
                               target_size=(224, 224),
                               interpolation="bicubic")
    img_arr = keras.utils.img_to_array(img=img)
    return np.array(img_arr)

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
else:
    print("Loading Saved Model from File")
    model = models.load_model(f'./trained_model.{save_format}')
print("Model finished training/loading")

# while(True):
#     img = input()
#     if img == "":
#         break
#     try:
#         filename = os.listdir(f'{data_path}/Test')[int(img)]
#     except:
#         continue
#     print(filename)
#     single_img = load_single_image(f'{data_path}/Test/{filename}')
#     result = model.predict(tf.expand_dims(single_img, axis=0))
#     print(result)
#     res_class, proba, idx =  interpret_result(result)
#     print(f"Predicted class: {classes[res_class]} (class={res_class}, proba={proba:.3})")
#     fig = plt.subplot()
#     fig.axis("off")
#     fig.set_title(f"Predicted class: {classes[res_class]} (class={res_class}, proba={proba:.3})")
#     fig.imshow(single_img/255)
#     plt.show()