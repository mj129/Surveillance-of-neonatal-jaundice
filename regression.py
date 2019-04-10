from numpy import *
import tensorflow as tf
import numpy as np
import os, keras
from keras import optimizers, metrics
from keras.models import Model
from keras.layers import Dense, Lambda, convolutional, Conv2DTranspose, GlobalAveragePooling2D, concatenate, multiply
from keras.applications.resnet50 import ResNet50
from keras.callbacks import TensorBoard
import keras.backend.tensorflow_backend as KTF
import pandas as pd
import utils
import keras.backend as T

# Constant
BATCH_SIZE = 32

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

# load data.
Dir = '/home/mj/data/hd/image'
# read the juandice value as label.
fo = pd.read_csv("/home/mj/data/hd/table/train_0708.csv")
fe = pd.read_csv("/home/mj/data/hd/table/test_1202.csv")
ftrain = fo.values
fval = fe.values

# load the resnet50 model as a basemodel
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
conv_end = base_model.get_layer(name='activation_49')
deconv1 = Conv2DTranspose(512, [2, 2], strides=(2, 2), activation='relu')(conv_end.output)
conv_4 = base_model.get_layer(name='activation_40')
deconv1_con = concatenate([conv_4.output, deconv1], axis=-1)

deconv2 = Conv2DTranspose(512, [2, 2], strides=(2, 2), activation='relu')(deconv1_con)
conv_3 = base_model.get_layer(name='activation_22')
deconv2_con = concatenate([conv_3.output, deconv2], axis=-1)

conv_end_2 = convolutional.Conv2D(1, (2, 2), padding='same', activation='relu', name='conv_end2')(deconv2_con)

deconv2_con_att = keras.layers.multiply([deconv2_con, conv_end_2])
deconv2_glo = GlobalAveragePooling2D()(deconv2_con_att)
# define the classifion layer
pred_fc1 = Dense(1024, activation='sigmoid', name='cls_fc1')(deconv2_glo)
pred_cls = Dense(3, activation='softmax', name='cls_output')(pred_fc1)
# define the regression layer
fc1 = Dense(1024, activation='sigmoid', name='reg_fc1')(deconv2_glo)
pred_reg = Dense(1, activation='sigmoid', name='reg_output')(fc1)

model = Model(inputs=base_model.input, outputs=[pred_cls, pred_reg, conv_end_2, pred_fc1])
adam = optimizers.Adam(lr=0.0001)
tr_imgs, pred_mask, bin_label, bil_value, id_values, bin_values = utils.next_batch(ftrain, Dir, BATCH_SIZE)
imgset_loss = utils.loss_imgset(id_values,BATCH_SIZE)
model.compile(optimizer=adam, loss=['categorical_crossentropy', 'mean_absolute_error', 'binary_crossentropy', imgset_loss],
              loss_weights=[1., 1., 0.1, 0.1], metrics={'cls_output': 'acc'})

callback = TensorBoard('./log')
callback.set_model(model)
train_names = ['train_loss', 'cls_loss', 'reg_loss', 'seg_loss', 'imgset_loss', 'cls_acc']
val_names = ['val_loss', 'val_cls_loss', 'val_reg_loss', 'val_seg_loss', 'val_imgset_loss', 'val_cls_acc']

i = 0
while True:
    tr_imgs, pred_mask, bin_label, bil_value, id_values, bin_values = utils.next_batch(ftrain, Dir, BATCH_SIZE)
    # train the model
    Logs = model.train_on_batch(tr_imgs, [bin_label, bil_value, pred_mask, id_values])
    utils.write_log(callback, train_names, Logs, i)
    print("num=%i, train total loss=%.6f, cls loss=%.6f, mae=%.6f, seg=%.6f, imgset=%.6f, acc=%.5f" % (i, Logs[0], Logs[1], Logs[2], Logs[3], Logs[4], Logs[5]))
    if i % 100 == 0:
        # validate
        val_imgs, val_pred_mask, val_bin_label, val_bil_value, id_values, val_bin_values = utils.next_batch_test\
            (fval, Dir, batch_size = fval.shape[0]//BATCH_SIZE*BATCH_SIZE)
        Logs_val  = model.evaluate(val_imgs, [val_bin_label, val_bil_value, val_pred_mask, id_values], batch_size=BATCH_SIZE)
        utils.write_log(callback, val_names, Logs_val, i)
        print("val total loss=%.6f, cls loss=%.6f, mae=%.6f, seg=%.6f, imgset=%.6f, acc=%.5f" % (Logs_val[0], Logs_val[1], Logs_val[2], Logs_val[3], Logs_val[4], Logs[5]))
    if i % 100 == 0:
        # save model
        model.save('./save/bincls_reg_%d' % i + '.h5')
    i += 1
