# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:33:18 2024

@author: danielaugusto
"""

import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import shutil
import random
#from tqdm.notebook import tqdm
import os
#import PIL
'''from PIL import Image
import pandas as pd
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import cv2
from osgeo import osr

import geopandas as gpd
#from shapely.geometry import Polygon
from shapely.geometry import box'''

#%%
# Organiza pastas

source_folder = "C:\\treino_2"
train_path_img = 'C:\\treino_2\\images\\train'
train_path_labels = 'C:\\treino_2\\labels\\train'
val_path_img = 'C:\\Treino\\images\\val'
val_path_labels = 'C:\\treino_2\\labels\\val'
results_folder = "C:\\treino_2\\resultados"

#%%

# Organiza arquivos em treino e teste

# Define the split ratio (e.g., 80% for training, 20% for testing)
split_ratio = 0.8

# Create destination folders if they don't exist
os.makedirs(train_path_img, exist_ok=True)
os.makedirs(val_path_img, exist_ok=True)
os.makedirs(train_path_labels, exist_ok=True)
os.makedirs(val_path_labels, exist_ok=True)

# Get a list of all .jpg files in the source folder
jpg_files = [f for f in os.listdir(source_folder) if f.endswith(".png")]

# Randomly shuffle the list of .jpg files
random.shuffle(jpg_files)

# Calculate the number of files for the training set
split_index = int(len(jpg_files) * split_ratio)

# Split the files into training and test sets
train_jpg_files = jpg_files[:split_index]
test_jpg_files = jpg_files[split_index:]

# Move .jpg files to their respective folders
for jpg_file in train_jpg_files:
    shutil.copy2(os.path.join(source_folder, jpg_file), os.path.join(train_path_img, jpg_file))
for jpg_file in test_jpg_files:
    shutil.copy2(os.path.join(source_folder, jpg_file), os.path.join(val_path_img, jpg_file))

# Move corresponding .txt files to their respective label folders
for jpg_file in train_jpg_files:
    txt_file = jpg_file.replace(".png", ".txt")
    shutil.copy2(os.path.join(source_folder, txt_file), os.path.join(train_path_labels, txt_file))
for jpg_file in test_jpg_files:
    txt_file = jpg_file.replace(".png", ".txt")
    shutil.copy2(os.path.join(source_folder, txt_file), os.path.join(val_path_labels, txt_file))

#%%
# Carrega modelo pré-treinado
model = YOLO("yolov10x.pt")  # load a pretrained model (recommended for training)

y = "C:\\dataset.yaml"

#%%

# Use the model

results = model.train(data = y, #define os parametros do conjunto de dados
                      epochs=300, #épocas de treino
                      imgsz=640, #tamanho das imagens de treino px
                      batch=10, #imagens por batch de treino
                      augment=True, #aplicar aumentação
                      project=results_folder, #onde salvar
                      single_cls = True, #True = foca na detecção de objeto, não na classe do objeto
                      optimizer = 'auto', #Escolhe o otimizador
                      amp = True, #Automatic Mixed Precision
                      label_smoothing = 0.1, #suavisação de rótulos 
                      plots = True, #Plota gráficos do treino
                      name= 'especies_v10_single')


#%%

# Test in validation images

best = "C:\\treino_2\\resultados\\especies_v10_multi300\\weights\\best.pt"
#save_dir = "C:\\treino_2\\resultados\\val"


# Teste com imagens JPG
model.predict(model=best, 
              conf=0.45,
              iou=0.7,
              source=val_path_img, 
              save = True, 
              show_labels = True, 
              show_conf = False, 
              max_det = 600, 
              agnostic_nms = True, 
              augment = False 
              ) 
