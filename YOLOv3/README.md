# YOLO_v3_tutorial_from_scratch
Accompanying code for Paperspace tutorial series ["How to Implement YOLO v3 Object Detector from Scratch"](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)
This is where the majority of the code was taken from.


## Source code for the sprite sheet generator explained in
https://minzkraut.com/2016/11/23/making-a-simple-spritesheet-generator-in-python/


# The underlying architectyre of YOLO and contains code that creates the YOLO network
darknet.py


# Code for various helper functions
util.py


# The main algorithm which also has the sprite-sheet generator integrated
detect.py


# Folder in which images are kept to be analyzed
imgs folder

# Folder ready images (with bounding boxes) are stored
det folder

# Folder with extra images (previusly used in imgs folder)
extra_images folder

# Folder in which all sprites are located
frames folder


