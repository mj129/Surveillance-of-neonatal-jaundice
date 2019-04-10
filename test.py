import numpy as np
import pandas as pd
import cv2, os
from keras.models import load_model
import utils

# Constant
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
BATCH_SIZE = 12

# load data
Dir = './imgs'
fo = pd.read_csv('./imgs/labels.csv')
ftest = fo.values
te_num = ftest.shape[0]

# test
images, labels, bvalues, id_values, imgname = utils.batch_test(ftest, Dir, batch_size = BATCH_SIZE)

# load model
model = load_model('./model/model.h5', custom_objects={'loss_imgset_fn': utils.loss_imgset(id_values, BATCH_SIZE)})
results = model.predict(images, batch_size=BATCH_SIZE, verbose=1)
pbilvalues = np.squeeze(results[1])
dataframe = pd.DataFrame({'ID':id_values, 'imgname':imgname, 'pbil_value':pbilvalues})
dataframe.to_csv('./res.csv')
print("done")
