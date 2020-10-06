from layers import *
from Stackker import Sequential
import json
import numpy as np
def load_model(path="model.json"):
    """
        path:- path of model file including filename        
        returns:- a model
    """
    
    models = {"Sequential": Sequential}
    #layers = {"FFL": FFL}
    layers = {"FFL": FFL, "Conv2d":Conv2d, "Dropout":Dropout, "Flatten": Flatten, "Pool2d":Pool2d}
    with open(path, "r") as f:
        dict_model = json.load(f)
        model = dict_model["model"]
        #exec("model=models[model]")
        model = models[model]()
        #exec("model=model()")
        for layer, params in dict_model.items():
            if layer != "model":
                # create a layer obj
                lyr_type = layers[params["type"]]
                #print(layers[params["type"]], Conv2d)
                
                ###### create models here.                
                if lyr_type == FFL:                                        
                    lyr.neurons = params["neurons"]
                    lyr = layers[params["type"]](neurons=params["neurons"])
                
                if lyr_type == Conv2d:
                    lyr = layers[params["type"]](filters=int(params["filters"]), kernel_size=params["kernel_size"], padding=params["padding"])
                    #print(params["output_shape"])
                    lyr.out = np.zeros(params["output_shape"])
                    params["input_shape"] = tuple(params["input_shape"])
                    params["output_shape"] = tuple(params["output_shape"])
                if lyr_type == Dropout:
                    lyr = layers[params["type"]](prob=params["prob"])
                    try:
                        params["input_shape"] = tuple(params["input_shape"])
                        params["output_shape"] = tuple(params["output_shape"])
                    except:
                        pass
                    
                if lyr_type == Pool2d:
                    lyr = layers[params["type"]](kernel_size = params["kernel_size"], stride=params["stride"], kind=params["kind"])
                    params["input_shape"] = tuple(params["input_shape"])
                    try:
                        params["output_shape"] = tuple(params["output_shape"])
                    except:
                        pass
                if lyr_type == Flatten:
                    params["input_shape"] = tuple(params["input_shape"])                    
                    lyr = layers[params["type"]](input_shape=params["input_shape"])
                lyr.name = layer
                lyr.activation = params["activation"]
                lyr.isbias = params["isbias"]
                lyr.input_shape = params["input_shape"]
                lyr.output_shape = params["output_shape"]
                lyr.parameters = int(params["parameters"])
                
                if params.get("weights"):
                    lyr.weights = np.array(params["weights"])
                
                if params.get("biases"):
                    lyr.biases = np.array(params["biases"])               
                
                model.layers.append(lyr)
        print("Model Loaded...")
        return model