# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:28:40 2024

@author: danielaugusto
"""
# Carrega pacotes

import ultralytics
ultralytics.checks()
from ultralytics import YOLO
#import shutil
import random
#from tqdm.notebook import tqdm
import os
#import PIL
from PIL import Image
import pandas as pd
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import cv2
from osgeo import osr

import geopandas as gpd
#from shapely.geometry import Polygon
#from shapely.geometry import box
from shapely.geometry import Polygon

#%%

# Pastas com arquivos png/jpg e modelos

files = 'C:\\images'

model = YOLO("C:\\treino_2\\resultados\\especies_v10_multi300\\weights\\best.pt")

best = "C:\\treino_2\\resultados\\especies_v10_multi300\\weights\\best.pt"

#%%

# Aplica modelo em pngs
model.predict(model=best, 
              conf=0.3,
              iou=0.9,
              imgsz=4864,
              source=files, 
              save = True, 
              show_labels = True, 
              show_conf = False, 
              max_det = 600, 
              agnostic_nms = True, 
              augment = False 
              ) 

#%%

# Carrega ortofoto, pega coordenadas de cada pixel e constroi a imagem bgr

def scaleMinMax(x):
    return((x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)))

# Teste com ortofoto

ds = gdal.Open('C:\\Users\\danielaugusto\\Documents\\Daniel\\ortos\\ortofotos\\orto_RGB_974.tif')


# Define source and target projections
# Get the geotransform and projection
geotransform = ds.GetGeoTransform()
source_srs = osr.SpatialReference()
source_srs.ImportFromWkt(ds.GetProjection())

target_srs = osr.SpatialReference()
target_srs.ImportFromEPSG(32722)  # WGS 84 / UTM zone 22S

# Create the coordinate transformation
transform = osr.CoordinateTransformation(source_srs, target_srs)

# Perform the reprojection in memory
mem_drv = gdal.GetDriverByName('MEM')
ds = gdal.Warp(
    '', ds,
    format='MEM',  # Use the in-memory driver
    dstSRS=target_srs.ExportToWkt(),
    resampleAlg='bilinear'
)


r = ds.GetRasterBand(1).ReadAsArray()
g = ds.GetRasterBand(2).ReadAsArray()
b = ds.GetRasterBand(3).ReadAsArray()

r = scaleMinMax(r)
g = scaleMinMax(g)
b = scaleMinMax(b)

large_image = np.dstack((r,g,b)) #OpenCV standard

rgb = (large_image*255).astype(np.uint8)

plt.imshow(rgb)

img = Image.fromarray(rgb)

plt.imshow(img)

#%%
'''
results = model.predict(img,conf=0.45,
                        iou=0.7,
                        imgsz=12341,
                        agnostic_nms = True,
                        save = False)


'''
#%%
# Recorta raster original em pedaços de 640x640 mantendo as coordenadas originais do pixel para rodar o modelo de detecção

def get_coordinates(x, y, geotransform):
    """
    Calculates the geographic coordinates of the upper-left corner of the pixel (x, y)
    based on the image's GeoTransform.
    """
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    
    coordX = originX + (x * pixelWidth)
    coordY = originY + (y * pixelHeight)
    
    return coordX, coordY


def split_image_with_coordinates(image, geotransform, block_size=3000, step_size=1500):
    """
    Splits the image into blocks of `block_size`x`block_size` and returns each block 
    with its XY coordinates, moving with a step size of `step_size`.
    """
    img_height, img_width, _ = image.shape
    blocks = []
    
    for y in range(0, img_height - block_size + 1, step_size):
        for x in range(0, img_width - block_size + 1, step_size):
            # Crop a block of block_size x block_size from the image
            block = image[y:y+block_size, x:x+block_size, :]
            
            # Get coordinates of the upper-left corner of the block
            coordX, coordY = get_coordinates(x, y, geotransform)
            
            # Add the block and its coordinates
            blocks.append((block, (coordX, coordY)))
    
    return blocks

# Example: Split the image with a 640x640 block size and 320x320 step
blocks_with_coords = split_image_with_coordinates(rgb, geotransform, block_size=3000, step_size=1500)

# Accessing a block and its coordinates
for block, (coordX, coordY) in blocks_with_coords:
    print(f"Block at coordinates (X={coordX}, Y={coordY})")
    
    
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]   

#%%
# Randomly select an item from blocks_with_coords
random_block, (coordX, coordY) = random.choice(blocks_with_coords)

# Plot the random block
plt.imshow(random_block)
plt.title(f"Block at Coordinates (X={coordX}, Y={coordY})")
#plt.axis('off')  # Hide axes
plt.show()
'''
img_height, img_width, _ = rgb.shape
for y in range(0, img_width, 640):
    print(y)
'''
#%%

# Lista para armazenar DataFrames de todos os blocos
all_detections = []

# Aplica o modelo YOLOv10 em cada bloco de imagem
for block, (coordX, coordY) in blocks_with_coords:
    # Salva temporariamente o bloco como imagem e aplica o modelo
    #cv2.imwrite('temp_block.jpg', block)  # Salva o bloco temporariamente para detecção
    results = model.predict(block,conf=0.35, iou=0.5, imgsz=3000, agnostic_nms = True, save = False)  # Aplica o modelo YOLOv10 no bloco

    # Para cada detecção no bloco
    for result in results:
        # Obtem as coordenadas de detecção (caixas), classes e confiança
        boxx = result.boxes.xyxy
        con = result.boxes.conf
        classes = result.boxes.cls

        # Converte para DataFrames
        boxx = boxx.cpu()
        pbox = pd.DataFrame(boxx.numpy())
        pbox = pbox.rename(columns={0: "xmin", 1: "ymin", 2: "xmax", 3: "ymax"})

        con = con.cpu()
        pcon = pd.DataFrame(con.numpy())
        pcon = pcon.rename(columns={0: "confidence"})

        classes = classes.cpu()
        pclasses = pd.DataFrame(classes.numpy())
        pclasses = pclasses.rename(columns={0: "class"})

        # Ajusta as coordenadas com base na posição do bloco (coordX, coordY)
        pbox["xmin"] = (pbox["xmin"] * pixelWidth) + coordX
        pbox["ymin"] = (pbox["ymin"] * pixelHeight) + coordY
        pbox["xmax"] = (pbox["xmax"] * pixelWidth) + coordX
        pbox["ymax"] = (pbox["ymax"] * pixelHeight) + coordY

        # Junta tudo em um DataFrame só para este bloco
        pbox = pbox.join(pcon)
        pbox = pbox.join(pclasses)

        # Adiciona o DataFrame deste bloco à lista geral
        all_detections.append(pbox)

# Concatena todos os DataFrames de todos os blocos em um único DataFrame final
final_detections = pd.concat(all_detections, ignore_index=True)

# Agora você tem todas as detecções com coordenadas e classes em final_detections
print(final_detections)


#%%

# Salva o dataframe como csv no computador
folder_path = 'C:\\'
filename = 'detect_multi_974.csv'
file_path = os.path.join(folder_path, filename)

# Save the DataFrame as an Excel file
final_detections.to_csv(file_path, index=False)

print(f"DataFrame saved to {file_path}")

#%%

# Função para criar um polígono (retângulo) com base nas coordenadas xmin, ymin, xmax, ymax
def create_rectangle(row):
    xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
    return Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

# Cria uma coluna 'geometry' no DataFrame com os polígonos (retângulos)
final_detections['geometry'] = final_detections.apply(create_rectangle, axis=1)

# Converte o DataFrame pandas em um GeoDataFrame do geopandas
gdf = gpd.GeoDataFrame(final_detections, geometry='geometry')

# Define o sistema de coordenadas (por exemplo, EPSG:32722 para WGS 84 / UTM zone 22S)
gdf.set_crs(epsg=32722, inplace=True)

# Salva o GeoDataFrame como um shapefile
gdf.to_file("C:\\detected_multi_974.shp", 
            driver='ESRI Shapefile')

print("Shapefile salvo com sucesso!")


