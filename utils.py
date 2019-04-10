from numpy import *
import numpy as np
import random, os
import cv2
import tensorflow as tf
import math
import keras.backend as T

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def loss_imgset(id_values,BATCH_SIZE, K2=6):
    def loss_imgset_fn(y_true, y_pred):
        Kintra_list = [[] for i in range(BATCH_SIZE)]
        Kinter_list = [[] for i in range(BATCH_SIZE)]
        H = 0
        # id_values = y_true
        for i in range(BATCH_SIZE):
            ID_i = id_values[i]
            for j in range(i + 1, BATCH_SIZE):
                ID_j = id_values[j]
                y_pred_i = T.expand_dims(y_pred[i, :], axis=1)
                y_pred_j = T.expand_dims(y_pred[j, :], axis=1)
                dij = tf.matmul(T.transpose(y_pred_i - y_pred_j), (y_pred_i - y_pred_j))
                dij = tf.squeeze(dij, axis=1)
                dji = dij
                if ID_j == ID_i:
                    Kintra_list[i].append(dij)
                    Kintra_list[j].append(dji)
                else:
                    Kinter_list[i].append(dij)
                    Kinter_list[j].append(dji)
            if Kintra_list[i]:
                D1 = tf.reduce_mean(tf.cast(Kintra_list[i], tf.float32), axis=0)
            else:
                D1 = 0
            D2 = tf.reduce_mean(tf.nn.top_k(-tf.squeeze(tf.cast(Kinter_list[i], tf.float32), axis=1), K2)[0], axis=0)
            hi = T.relu(D1 + 0.5 * D2)
            H += hi
        return H / BATCH_SIZE
    return loss_imgset_fn

def select_id(DATA, batch_size):
    la = np.unique(DATA[:, 1])
    per = np.arange(la.shape[0])
    np.random.shuffle(per)
    perm = []
    # perm1 = []
    i = 0
    while len(perm) < batch_size:
        sid=la[per[i]]
        dataid=np.argwhere(DATA[:,1]==sid)
        if dataid.shape[0]<3:
            for j in range(dataid.shape[0]):
                if len(perm) < batch_size:
                    perm.append(dataid[j,0])

        else:
            daper=np.arange(dataid.shape[0])
            np.random.shuffle(daper)
            for j in range(0,2):
                if len(perm) < batch_size:
                    perm.append(dataid[daper[j],0])

        # for j in range(DATA.shape[0]):
        #     if DATA[j,1] == sid and len(perm1) < batch_size:
        #         perm1.append(j)
        i += 1
    perm = np.asarray(perm)
    np.random.shuffle(perm)
    return perm

def next_batch(DATA, Dir, batch_size):
    perm = select_id(DATA, batch_size)
    # perm1 = np.arange(len(DATA))
    # np.random.shuffle(perm1)
    id_values = np.zeros([batch_size,])
    bin_values = np.zeros([batch_size, ])
    bvalues = np.zeros([batch_size,])
    labels = np.zeros([batch_size, 3])
    images = np.zeros([batch_size,224,224,3])
    pred_masks = np.zeros([batch_size, 28, 28, 1])
    count = 0
    for i in perm[:batch_size]:
        bvalue = DATA[i, :][5]
        fname = DATA[i, :][8]
        label = DATA[i,:][3]
        id_value = DATA[i, :][1]
        bin_value = DATA[i, :][3]
        img_path = Dir +'/' + fname
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pred_mask = cv2.imread(Dir + '-segmentation/' + os.path.splitext(fname)[0] + '_mask.jpg', 0)
        pred_mask = np.rint(pred_mask / 255)
        image = cv2.resize(image, dsize=(224, 224))
        pred_mask = cv2.resize(pred_mask, dsize=(28, 28))
        pred_mask = np.expand_dims(pred_mask, axis=2)
        if bvalue * 400 / 17.1 <= 5:
            labels[count, :] = [1, 0, 0]
        elif bvalue * 400 / 17.1 > 5 and bvalue * 400 / 17.1 < 15:
            labels[count, :] = [0, 1, 0]
        else:
            labels[count, :] = [0, 0, 1]
        id_values[count] = id_value
        bin_values[count] = bin_value
        bvalues[count] = bvalue
        images[count,:,:,:] = image
        pred_masks[count,:,:,:] = pred_mask
        count += 1
    return images, pred_masks, labels, bvalues, id_values, bin_values

def next_batch_test(DATA, Dir, batch_size):
    perm = np.arange(len(DATA))
    # np.random.shuffle(perm)
    id_values = np.zeros([batch_size,])
    bin_values = np.zeros([batch_size, ])
    bvalues = np.zeros([batch_size,])
    labels = np.zeros([batch_size, 3])
    images = np.zeros([batch_size,224,224,3])
    pred_masks = np.zeros([batch_size, 28, 28, 1])
    count = 0
    for i in perm[:batch_size]:
        bvalue = DATA[i, :][5]
        fname = DATA[i, :][8]
        label = DATA[i,:][3]
        id_value = DATA[i, :][1]
        bin_value = DATA[i, :][3]
        img_path = Dir + '/' + fname
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred_mask = cv2.imread(Dir + '-segmentation/' + os.path.splitext(fname)[0] + '_mask.jpg', 0)
        pred_mask = np.rint(pred_mask / 255)
        # if np.max(pred_mask) == 1:
        #     image = pred_mask * image
        # else:
        #     image = image
        image = cv2.resize(image, dsize=(224, 224))
        pred_mask = cv2.resize(pred_mask, dsize=(28, 28))
        pred_mask = np.expand_dims(pred_mask, axis=2)
        #img, pred_mask = hp(image)

        if bvalue * 400 / 17.1 <= 5:
            labels[count, :] = [1, 0, 0]
        elif bvalue * 400 / 17.1 > 5 and bvalue * 400 / 17.1 < 15:
            labels[count, :] = [0, 1, 0]
        else:
            labels[count, :] = [0, 0, 1]
        id_values[count] = id_value
        bin_values[count] = bin_value
        bvalues[count] = bvalue
        images[count,:,:,:] = image
        pred_masks[count, :, :, :] = pred_mask
        count+=1
    return images, pred_masks, labels, bvalues, id_values, bin_values

def batch_test(DATA, Dir, batch_size):
    perm = np.arange(len(DATA))
    id_values = np.zeros([batch_size, ])
    bvalues = np.zeros([batch_size, ])
    labels = np.zeros([batch_size, 3])
    images = np.zeros([batch_size, 224, 224, 3])
    imgname = []
    count = 0
    for i in perm[:batch_size]:
        bvalue = DATA[i, :][3]
        fname = DATA[i, :][4]
        id_value = DATA[i, :][1]
        img_path = Dir + '/' + fname
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(224, 224))

        if bvalue * 400 / 17.1 <= 5:
            labels[count, :] = [1, 0, 0]
        elif bvalue * 400 / 17.1 > 5 and bvalue * 400 / 17.1 < 15:
            labels[count, :] = [0, 1, 0]
        else:
            labels[count, :] = [0, 0, 1]
        id_values[count] = id_value
        bvalues[count] = bvalue
        imgname.append(fname)
        images[count, :, :, :] = image
        count += 1
    return images, labels, bvalues, id_values, imgname

def randnorepeat(m,n):
    p=list(range(n))
    d=random.sample(p,m)
    return d

def train_val(fo, validate_rate = 0.1):
    la = np.unique(fo.values[:, 1])

    darange = mat(zeros((2, 2)), dtype=int)
    sum_all, sum_1, sum_0 = la.shape[0], np.sum(la > 1000), np.sum(la < 1000)
    darange[0,0], darange[0,1], darange[1,0], darange[1,1] = 1000 + 1, 1000 + sum_1, 1, sum_0

    train_data = []
    test_data = []
    for i in range(2):
        ran1 = darange[i, 0]
        ran2 = darange[i, 1]
        category_num = ran2 - ran1 + 1
        num_train_sample = int(round(category_num * (1 - validate_rate)))
        train_data_ele = ran1 + randnorepeat(num_train_sample, category_num)
        ran_a = ran1 + list(range(category_num))
        train_data.append(train_data_ele)
        test_data.append([l for l in ran_a if l not in train_data_ele])

    index_train = np.asarray(train_data[0:1])
    index_test = np.asarray(test_data[0:1])
    tem = np.asarray(train_data[1:2])
    index_train = np.concatenate((index_train, tem), axis=1)
    te = np.asarray(test_data[1:2])
    index_test = np.concatenate((index_test, te), axis=1)
    index_train = index_train[0, :]
    index_test = index_test[0, :]

    in_train = np.argwhere(fo.values[:, 1] == index_train[0])
    for i in range(1, index_train.shape[0]):
        in_train_tem = np.argwhere(fo.values[:, 1] == index_train[i])
        in_train = np.concatenate((in_train, in_train_tem), axis=0)
    in_val = np.argwhere(fo.values[:, 1] == index_test[0])
    for i in range(1, index_test.shape[0]):
        in_val_tem = np.argwhere(fo.values[:, 1] == index_test[i])
        in_val = np.concatenate((in_val, in_val_tem), axis=0)

    fo_train = fo.values[in_train, :]
    fo_train = np.reshape(fo_train, [fo_train.shape[0], fo_train.shape[2]])
    fo_val = fo.values[in_val, :]
    fo_val = np.reshape(fo_val, [fo_val.shape[0], fo_val.shape[2]])
    return fo_train, fo_val

def train_test(fo,fe):
    a_id = fo.values[:,1]
    te_id = fe.values[:,0]
    idx_list = []
    for id in te_id:
        idx = np.where(a_id == id)
        idx = list(np.squeeze(np.asarray(idx)))
        idx_list.extend(idx)
    ftest = fo.values[idx_list,:]
    idx_all = fo.values[:,0]
    idx_train = np.setdiff1d(idx_all, np.asarray(idx_list))
    idx_train = idx_train.astype(int)
    ftrain = fo.values[idx_train,:]
    return ftrain, ftest

def train_val_multi(fo, validate_rate = 0.1):
    # la = np.unique(fo.values[:, 1])
    # darange = mat(zeros((2, 2)), dtype=int)
    # sum_all, sum_1, sum_0 = la.shape[0], np.sum(la > 1000), np.sum(la < 1000)
    # darange[0,0], darange[0,1], darange[1,0], darange[1,1] = 1000 + 1, 1000 + sum_1, 1, sum_0

    num_g = 30
    num_all = 0
    darange = mat(zeros((num_g, 2)), dtype=int)
    for i in range(num_g):
        wi = np.argwhere(fo.values[:, 4] == i)
        lai = fo.values[wi, :]
        lai = np.reshape(lai, [lai.shape[0], lai.shape[2]])
        la = np.unique(lai[:, 1])
        num_la = la.shape[0]
        darange[i,0] = num_all + 1
        darange[i,1] = num_all + num_la
        num_all = num_all + num_la

    train_data = []
    test_data = []
    for i in range(num_g):
        wi = np.argwhere(fo.values[:, 4] == i)
        lai = fo.values[wi, :]
        lai = np.reshape(lai, [lai.shape[0], lai.shape[2]])
        la = np.unique(lai[:, 1])

        ran1 = darange[i, 0]
        ran2 = darange[i, 1]
        category_num = ran2 - ran1 + 1
        num_train_sample = int(category_num * (1 - validate_rate))
        train_data_ele = la[randnorepeat(num_train_sample, category_num)]
        # train_data_ele = ran1 + randnorepeat(num_train_sample, category_num)
        # ran_a = ran1 + list(range(category_num))
        ran_a = la
        train_data.append(train_data_ele)
        test_data.append([l for l in ran_a if l not in train_data_ele])

    index_train = np.asarray(train_data[0:1])
    index_test = np.asarray(test_data[0:1])
    for i in range(1, num_g):
        tem = np.asarray(train_data[i:i + 1])
        index_train = np.concatenate((index_train, tem), axis=1)

        te = np.asarray(test_data[i:i + 1])
        index_test = np.concatenate((index_test, te), axis=1)

    index_train = index_train[0, :]
    index_test = index_test[0, :]

    in_train = np.argwhere(fo.values[:, 1] == index_train[0])
    for i in range(1, index_train.shape[0]):
        in_train_tem = np.argwhere(fo.values[:, 1] == index_train[i])
        in_train = np.concatenate((in_train, in_train_tem), axis=0)
    in_val = np.argwhere(fo.values[:, 1] == index_test[0])
    for i in range(1, index_test.shape[0]):
        in_val_tem = np.argwhere(fo.values[:, 1] == index_test[i])
        in_val = np.concatenate((in_val, in_val_tem), axis=0)

    fo_train = fo.values[in_train, :]
    fo_train = np.reshape(fo_train, [fo_train.shape[0], fo_train.shape[2]])
    fo_val = fo.values[in_val, :]
    fo_val = np.reshape(fo_val, [fo_val.shape[0], fo_val.shape[2]])
    return fo_train, fo_val