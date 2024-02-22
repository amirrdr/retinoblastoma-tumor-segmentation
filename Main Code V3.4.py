import glob
import imageio
import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Model
from keras.callbacks import CSVLogger
from keras.layers import *
from sklearn.model_selection import train_test_split

shape = (240, 320, 3)
TS = 0.1
DO = 0.1
KS_1 = 32
KS_2 = 8
KS_3 = 2
KS = 8
EP = 20
VS = 0.1
BS = 1
LR = 0.001


def set_to_white(image):
    h = image.shape[0]
    w = image.shape[1]
    for x in range(h):
        for y in range(w):
            if np.sqrt(((x - (h/2)) ** 2) + ((y - (w/2)) ** 2)) > (0.97 * w/2):
                image[x, y] = 1
    return image


class PerformancePlotCallback(keras.callbacks.Callback):
    def __init__(self, x_test, model_name):
        self.x_test = x_test
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_test)
        for k in range(len(y_pred)):
            y_pred[k] = 1 - set_to_white(y_pred[k])
            y_pred[k] = (y_pred[k] - np.min(y_pred[k])) / np.ptp(y_pred[k])
            imageio.imwrite('./Outputs/Main Code V3.4/Segmentations by Model/Epoch Based/a ('
                            + str(epoch + 1) + ')/' + str(k) + ' - Segments.JPG',
                            np.array(255 * y_pred[k]).astype('uint8'))


inputs_directory = './Data/Inputs'
filelist = glob.glob(inputs_directory + '/*.*')
inputs = [np.array(Image.open(fname).resize((shape[1], shape[0])).__array__()) for fname in filelist]
inputs = np.array(inputs)

outputs_directory = './Data/Outputs'
filelist = glob.glob(outputs_directory + '/*.*')
outputs = [np.array(Image.open(fname).resize((shape[1], shape[0])).convert('L').__array__()) for fname in filelist]
outputs = np.array(outputs)
outputs[outputs < 128] = 0
outputs[outputs >= 128] = 1

print('Inputs Shape:', inputs.shape)
print('Outputs Shape:', outputs.shape)

inputs, inputs_evl, outputs, outputs_evl = train_test_split(inputs, outputs, test_size=TS, shuffle=True)

for i in range(inputs_evl.shape[0]):
    imageio.imwrite('./Outputs/Main Code V3.4/Original Images/' + str(i) + '.JPG', inputs_evl[i])

inp = Input(shape=shape)

ltn_1 = Conv2D(4, KS_1, activation='relu', padding='same')(inp)
ltn_2 = Conv2D(2, KS_2, activation='relu', padding='same')(inp)
ltn_3 = Conv2D(1, KS_3, activation='relu', padding='same')(inp)

ltn = Concatenate(axis=-1)([ltn_1, ltn_2, ltn_3])
ltn = BatchNormalization()(ltn)

ltn_1 = Conv2D(8, KS_1, activation='relu', padding='same')(ltn)
ltn_2 = Conv2D(4, KS_2, activation='relu', padding='same')(ltn)
ltn_3 = Conv2D(2, KS_3, activation='relu', padding='same')(ltn)

ltn = Concatenate(axis=-1)([ltn_1, ltn_2, ltn_3])
ltn = BatchNormalization()(ltn)

ltn_1 = Conv2D(16, KS_1, activation='relu', padding='same')(ltn)
ltn_2 = Conv2D(8, KS_2, activation='relu', padding='same')(ltn)
ltn_3 = Conv2D(4, KS_3, activation='relu', padding='same')(ltn)

ltn = Concatenate(axis=-1)([ltn_1, ltn_2, ltn_3])
ltn = BatchNormalization()(ltn)

ltn = Conv2D(4, KS, activation='relu', padding='same')(ltn)
ltn = Conv2D(2, KS, activation='relu', padding='same')(ltn)
out = Conv2D(1, KS, activation='sigmoid', padding='same')(ltn)

mdl = Model(inputs=inp, outputs=out)
mdl.summary()

opt = tf.keras.optimizers.RMSprop(learning_rate=LR)
mdl.compile(optimizer=opt, loss="binary_crossentropy")
log_dir = "./Logs/Main Code V3.4/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
directory = log_dir + 'loss.csv'
csv_logger = CSVLogger(directory, append=True, separator=',')
performance = PerformancePlotCallback(inputs_evl, model_name="mdl")

mdl.fit(inputs, outputs, shuffle=True, epochs=EP, callbacks=[tensorboard_callback, csv_logger, performance],
        batch_size=BS)
mdl.save('./Models/Main Code V3.4/')

results = mdl.evaluate(inputs_evl, outputs_evl, verbose=0)
print('Evaluation Result: ', results)

Y_pred = mdl.predict(inputs_evl)
for i in range(len(Y_pred)):
    Y_pred[i] = 1 - set_to_white(Y_pred[i])
    Y_pred[i] = (Y_pred[i] - np.min(Y_pred[i])) / np.ptp(Y_pred[i])

for i in range(Y_pred.shape[0]):
    imageio.imwrite('./Outputs/Main Code V3.4/Segmentations by Model/Final Epoch/' + str(i) +
                    ' - Segments.JPG', np.array(255 * Y_pred[i]).astype('uint8'))
