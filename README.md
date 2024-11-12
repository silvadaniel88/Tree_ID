This repository contains the implementation of a YOLOv10 model to perform object detection and identification of a seven 'morpho species' of trees from aerial photographs. 
The morpho species are contained in the classes.txt file. A example of images with labels are given in the example folder.
The tree_train.py file fine tune a yolov10x.pt model to identify individual tree crowns and the morpho species.  
The tree_apply.py file shows a implementation in a orthomosaic, and incorporate the extraction of the geographyc coordinates of the objects detected and the creation of a shapefile containing these objects.  
