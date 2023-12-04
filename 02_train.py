from architecture import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd
import argparse
import pickle
import os

import warnings
warnings.filterwarnings("ignore")



ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', default=1,
                help="Specify the number of epochs", type=int)
args = vars(ap.parse_args())

MODEL_NAME = 'LeNet5'
LEARNING_RATE = 0.001
EPOCHS = args['epochs']
BATCH_SIZE = 32
INPUT_SHAPE = (100, 100, 3)
TRAIN_DIR = 'images/train'
VALID_DIR = 'images/valid'
OUTPUT_DIR = r'./output'

def plot_hist(history, filename):
    hist = pd.DataFrame(history.history)
    hist['epochs'] = history.epoch

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Accuracy", "Loss"))

    fig.add_trace(go.Scatter(x=hist['epochs'], y=hist["accuracy"], name='train_accuracy',
                             mode='markers+lines', marker_color='#f29407'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epochs'], y=hist["val_accuracy"], name='valid_accuracy',
                             mode='markers+lines', marker_color='#0771f2'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epochs'], y=hist["loss"], name='train_loss',
                             mode='markers+lines', marker_color='#f29407'), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist['epochs'], y=hist["val_loss"], name='valid_loss',
                             mode='markers+lines', marker_color='#0771f2'), row=2, col=1)

    fig.update_xaxes(title_text="Number of epochs", row=1, col=1)
    fig.update_xaxes(title_text="Number of epochs", row=2, col=1)
    fig.update_xaxes(title_text="Accuracy", row=1, col=1)
    fig.update_xaxes(title_text="Loss", row=2, col=1)
    fig.update_layout(width=1400, height=1000, title=f"Metrics : {MODEL_NAME}")

    po.plot(fig, filename=filename, auto_open=False)

train_datagen = ImageDataGenerator(
    rotation_range = 30,
    rescale = 1. / 255.,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
    )

valid_datagen = ImageDataGenerator(rescale = 1. / 255.)

train_generator = train_datagen.flow_from_directory(
    directory = TRAIN_DIR,
    target_size = INPUT_SHAPE[:2],
    batch_size = BATCH_SIZE,
    class_mode = 'binary'
)

valid_generator = valid_datagen.flow_from_directory(
    directory = VALID_DIR,
    target_size = INPUT_SHAPE[:2],
    batch_size = BATCH_SIZE,
    class_mode = 'binary'
)

architectures = {MODEL_NAME : models.LeNet5}
architecture = architectures[MODEL_NAME](input_shape=INPUT_SHAPE)
model = architecture.build()

model.compile(
    optimizer = Adam(learning_rate=LEARNING_RATE),
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

model.summary()

dt = datetime.now().strftime("%d_%m_%Y_%H_%M")
filepath = os.path.join('output','model_' + dt +'.hdf5')
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy',save_best_only=True)

print("[INFO] The model is being trained...")

history=model.fit_generator(
    generator = train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data = valid_generator,
    validation_steps = valid_generator.samples // BATCH_SIZE,
    epochs = EPOCHS,
    callbacks = [checkpoint]
)

if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

print("[INFO] Exporting plots to html...")
filename = os.path.join('output', 'report_' + dt + '.html')
plot_hist(history,filename=filename)

print("[INFO] Exporting labels to file...")
with open(r'output\labels.pickle', 'wb') as file:
    file.write(pickle.dumps(train_generator.class_indices))
    
print("[INFO] Work is done!")  