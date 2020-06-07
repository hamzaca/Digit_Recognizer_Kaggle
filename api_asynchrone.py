#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 03:30:46 2020

@author: hamza
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from PIL import  Image
from keras.models import model_from_json




app = Flask(__name__)



print("----------------- ------------ --------  API START  ----------------------- ------------ ---------")





 
trained_model_name = "data/digits_avgPool_150_epochs"

json_file = open(trained_model_name +".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(trained_model_name +".h5")

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model._make_predict_function()
print("----------------- ------------ --------  Model Loaded   ----------------------- ------------ ---------")



def addWhitePixelsArround(file,new_size = (28, 28)):
    
    """ Add white pixels arround the image to have the shape 28x28, 
        and resize the image to the same shape if the image shape is bigger than 28x28.
    


        return :  - image_new : filan image with shape 28x28
    
    """
    with Image.open(file).convert('L') as image:
        old_size = image.size

        if new_size > old_size:
                  
            # changer le nombres de pixels de sorte d'entendre l'image vers la gauche, la droite et le haut de l'image.
            # etendre vers le bas en respectant la parité des nombres.
    
            if old_size[0] % 2 == 0:
                    add_left = add_right = (new_size[0] - old_size[0]) // 2
            else:
                    add_left = (new_size[0] - old_size[0]) // 2
                    add_right = ((new_size[0] - old_size[0]) // 2) + 1

            if old_size[1] % 2 == 0:
                    add_top = add_bottom = (new_size[1] - old_size[1]) // 2
            else:
                    add_top = (new_size[1] - old_size[1]) // 2
                    add_bottom = ((new_size[1] - old_size[1]) // 2) + 1

            left = 0 - add_left
            top = 0 - add_top
            right = old_size[0] + add_right
            bottom = old_size[1] + add_bottom
                
            im_inverse = 255 - np.array(image)
            im = Image.fromarray(im_inverse)
                
            # By default, the added pixels are black
            image_new = im.crop((left, top, right, bottom))
                
            image_new = (255 - np.array(image_new))/255.
        #  if the size of the image is bigger than 28x28 so resize it.
        else :
            image_new = np.array(image.resize((28,28),Image.ANTIALIAS))
            
        
        print("----------------- ------------ --------  Image resized  ----------------------- ------------ ---------")


        return image_new


@app.route('/',methods=['POST'])
def predict():
    
    data = request.get_json()
    
    path_image = data["image_path"]
    print("----------------- ------------ --------  image_path  ----------------------- ------------ ---------")
    
    print("image_path = ", path_image)
    
    # read image
    image =  addWhitePixelsArround(path_image)
        
    y = model.predict(image.reshape(-1,28, 28 , 1))
    output = np.argmax(y, axis = 1)
    
        
    print("----------------- ------------ --------  OUt  ----------------------- ------------ ---------")
    
    print("out = ", output)
    
    return  jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)