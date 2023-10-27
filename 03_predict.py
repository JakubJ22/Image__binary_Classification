from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd
import argparse
import os

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help="type of images: [train,valid,test]")
ap.add_argument('-m', '--model', required=False, help="path to model")
args = vars(ap.parse_args())

INPUT_SHAPE = (100,100,3)

datagen = ImageDataGenerator(rescale = 1. / 255.)

generator = datagen.flow_from_directory(
    directory = args['dataset'],
    target_size = (100,100),
    batch_size = 1,
    class_mode = 'binary',
    shuffle=False
)

print("[INFO] Loading the model...")

model = load_model(args['model'])

y_prob = model.predict_generator(generator)
y_prob = y_prob.ravel()

y_true = generator.classes

predictions = pd.DataFrame({'y_prob':y_prob,'y_true':y_true}, index=generator.filenames)
predictions['y_pred'] = predictions['y_prob'].apply(lambda x : 1 if x > 0.5 else 0)
predictions['is_incorrect'] = (predictions['y_true'] != predictions['y_pred']) * 1
errors = list(predictions[predictions['is_incorrect']== 1].index)

y_pred = predictions['y_pred'].values

print(f"Confusion Matrix:\n{confusion_matrix(y_true,y_pred)}")
print(f"Classification Report:\n{classification_report(y_true, y_pred, target_names=generator.class_indices.keys())}")
print(f"Model Accuracy:\n{accuracy_score(y_true,y_pred) * 100:.2f}%")

label_map = generator.class_indices
label_map = dict((v,k) for k,v in label_map.items())
predictions['class'] = predictions['y_pred'].apply(lambda x: label_map[x])

predictions.to_csv(r'output\predictions.csv')

print(f"[INFO] Number of incorrect classified images: {len(errors)}\n[INFO] File names:")
for error in errors:
    print(error)