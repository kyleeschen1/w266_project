#==================================================================
# Libraries
#==================================================================

import numpy as np
import pandas as pd
import json

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

from keras import Model
from keras import regularizers
from keras.layers import concatenate as k_cc
from keras import backend as K

from sklearn.metrics import classification_report, confusion_matrix

from utilities import *
from data_manager import *

#==================================================================
# Model Building
#==================================================================

def add_layer_for_dense(inputs, name, nodes, activation = "relu", dropout = 0.2):
    '''Add a dense layer with a size specified by nodes.'''
    
    init = initializers.glorot_normal()
    k_regs = regularizers.l1_l2(l1=1e-5, l2=1e-4)
    
    x = layers.Dense(nodes, 
                     activation = activation, 
                     kernel_initializer = init,
                     kernel_regularizer= k_regs)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    return x

def build_default_model(name, trD, trL, evD, evL):
    '''Builds and compiles a default model.'''
    
    # Initial layers
    inputs = layers.Input(shape= trD.shape[1])
    outputs = add_layer_for_dense(inputs, name, 1024)
    outputs = add_layer_for_dense(outputs, name, 256)
    
    # Add "partition layer" for easy splitting
    outputs = layers.Lambda(lambda x: x, name = "partition")(outputs)
    
    # Add final layer then compile
    outputs = add_layer_for_dense(outputs, name, 128)
    outputs = layers.Dense(trL.shape[1], name = "logits")(outputs)
    outputs = layers.Activation('softmax', name='softmax')(outputs)
    model = compile_model(name, inputs, outputs)
    
    # Add data as attributes
    model.trD_ = trD
    model.evD_ = evD
    model.trL_ = trL
    model.evL_ = evL
    model.status_ = "..."
    model.save_name_ = model.name_
    
    return model


def add_distillation_arm(model, temperature = 1):
    '''Builds and compiles a default model.'''
    
    # Get model name, inputs, outputs, and output size
    name = model.name_
    input_col_size = model.inputs[0].shape.as_list()[1]
    output_col_size = model.outputs[0].shape.as_list()[1]
    
    # Get outputs of model "partition" layer, then add distillation layer
    partition = model.get_layer("partition")
    outputs = add_layer_for_dense(partition.output, name, 128)
    distill = add_distillation_layers(name, 
                                      outputs, 
                                      temperature, 
                                      output_col_size)
    
    return keras.models.Model(name = name, 
                               inputs =  model.inputs, 
                               outputs = [model.outputs, distill])
    
    

def add_distillation_layers(name, outputs, temperature, output_col_size):
    '''Returns a set of layers for distillation learning.'''
    distill = layers.Dense(output_col_size, name = "distill_logits")(outputs)
    distill = layers.Activation('softmax',name = "distill_softmax")(distill)
    return distill   
    
#==================================================================
# Model Interaction
#==================================================================

def compile_model(name, inputs, outputs, 
                  loss = 'categorical_crossentropy', 
                  optimizer = 'adam',  
                  metrics = ['acc']):
    '''Compiles model. Set as function to preserve defaults.'''
    
    model = keras.models.Model(inputs = inputs, outputs = outputs, name = name)
    model.name_ = model.name
    
    model.compile(loss = loss,  
                  optimizer = optimizer, 
                  metrics = metrics)
    return model

def fit_model(model, epochs = 5):
    '''Fits model with training and validation data.'''
    banner("Running model {}...".format(model.name_), symbol = "=")
    model.fit(model.trD_, 
              model.trL_, 
              validation_data=[model.evD_, model.evL_], 
              batch_size = 32,
              epochs=epochs, 
              verbose=1)
    
def save_models(models, fp = "models/"):
    for model in models:
        model.save("{}{}".format(fp, model.save_name_))

#==================================================================
# Theory of Mind Helpers
#==================================================================

def train_student_model(level, student, teacher, temperature = 1, epochs = 5, outward = True):
    '''Trains on softened predictions of teacher.'''
    
    # Clone student model (resets weights)
    student = clone_model_without_weights(student, level, temperature)

    
    # set parameters depending on whether model is set to inward or outward
    if outward:
        col = "logits"
        key = "softmax"
        log = "extrospective"
    else:
        col = "distilled_logits"
        key = "distilled_softmax"
        log = "introspective"
        
    student.status_ = "{} thinks that {} thinks that {}".format(student.name_, teacher.name_, student.status_)
    student.save_name_ = "{}_temp_{}_tom_{}_{}".format(student.name_, temperature, level, log)
        
    # get softened training data
    trL = get_soft_predictions(teacher, teacher.trD_, temperature, col)
    evL = get_soft_predictions(teacher, teacher.evD_, temperature, col)
    student.trL_ = {"distill_softmax": trL, "softmax": student.trL_}
    student.evL_ = {"distill_softmax": evL, "softmax": student.evL_}

    # run student
    fit_model(student, epochs)
    
    # Format for the next round
    student.trL_ = student.trL_[key]
    student.evL_ = student.evL_[key]
    
    return student

def get_soft_predictions(teacher, data, temperature, col = "logits"):
    '''Computes softened predictions from teacher model on input data.'''
    temp = K.function([teacher.layers[0].input], [teacher.get_layer(col).output])
    preds = temp([data])[0] / temperature
    return np.exp(preds) / np.sum(np.exp(preds))

def clone_model_without_weights(m, level, temperature):
    '''Clones model while preserving specific attributes.'''
    
    n = keras.models.clone_model(m)
    n.set_weights(m.get_weights())
    
    if level == 1:
        n = add_distillation_arm(m, temperature)
        
    n = transfer_attributes(m, n)
    
    # Build and compile model using loss weights 
    n.compile(loss = "categorical_crossentropy", 
              optimizer = "adam",
              loss_weights = {"distill_softmax": temperature ** 2, "softmax": 1},
              metrics = ["acc"])
    return n


def transfer_attributes(m, n):
    n.name_ = m.name_
    m.save_name_ = m.save_name_
    n.trD_ = m.trD_
    n.evD_ = m.evD_
    n.trL_ = m.trL_
    n.evL_ = m.evL_
    n.status_ = m.status_
    return n

#==================================================================
# Theory of Mind Models
#==================================================================

def theory_of_mind_model(level, model_A, model_B, temperature, epochs, outward = True):
    '''Trains input models on each others predicted labels AND actual labels.'''
        
    banner("Running Theory of Mind Level {}".format(level), symbol = "*")
    
    new_model_A = train_student_model(level, model_A, model_B, temperature, epochs, outward)
    new_model_B = train_student_model(level, model_B, model_A, temperature, epochs, outward) 
    
    return new_model_A, new_model_B
        
def introspection_model(level, 
                        model,
                        temperature, 
                        epochs):   
    '''Trains model on its own copy.'''
    
    return train_student_model(level,
                               model, model, 
                               temperature, 
                               epochs) 
