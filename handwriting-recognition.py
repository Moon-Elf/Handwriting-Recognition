import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam

train = pd.read_csv('/kaggle/input/handwriting-recognition/written_name_train_v2.csv')
valid = pd.read_csv('/kaggle/input/handwriting-recognition/written_name_validation_v2.csv')

plt.figure(figsize=(10, 10))

for i in range(4):
    ax = plt.subplot(2, 2, i+1)
    img_dir = '/kaggle/input/handwriting-recognition/train_v2/train/'+train.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap = 'gray')
    plt.title(train.loc[i, 'IDENTITY'], fontsize=12)
    plt.axis('off')

plt.subplots_adjust(wspace=0.2, hspace=-0.8)

# NaN in the labels
print("Number of NaNs in train set      : ", train['IDENTITY'].isna().sum())
print("Number of NaNs in validation set : ", valid['IDENTITY'].isna().sum())

# Dropping the NaNs
train.dropna(axis=0, inplace=True)
valid.dropna(axis=0, inplace=True)

# Some of the images have label 'UNREADABLE'
unread = train[train['IDENTITY'] == 'UNREADABLE']
unread.reset_index(inplace = True, drop=True)

plt.figure(figsize=(15, 10))

for i in range(4):
    ax = plt.subplot(2, 2, i+1)
    img_dir = '/kaggle/input/handwriting-recognition/train_v2/train/'+unread.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap = 'gray')
    plt.title(unread.loc[i, 'IDENTITY'], fontsize=12)
    plt.axis('off')

plt.subplots_adjust(wspace=0.2, hspace=-0.8)

# Removing the Unreadable
train = train[train['IDENTITY'] != 'UNREADABLE']
valid = valid[valid['IDENTITY'] != 'UNREADABLE']

# Coverting lowercase to uppercase
train['IDENTITY'] = train['IDENTITY'].str.upper()
valid['IDENTITY'] = valid['IDENTITY'].str.upper()

# Resetting the index
train.reset_index(inplace = True, drop=True) 
valid.reset_index(inplace = True, drop=True)

# Fuction to preprocess the img
def preprocess(img):
    (h, w) = img.shape                                    # Getting the height & width of the image
    
    final_img = np.ones([64, 256])*255                    # Blank white image
    
    # crop    
    if h > 64:
        img = img[:64, :]                                 # If the h>64 then it is cropped to 64
        
    if w > 256:
        img = img[:, :256]                                # If the w>256 then it is cropped to 256
    
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE) # Rotate 90° Clockwise & return


train_size = 20000
valid_size = 2000

train_x = []                                                # Initialize a list to store preprocessed images

for i in range(train_size):                                 # Loop through the training dataset
    img_dir = '/kaggle/input/handwriting-recognition/train_v2/train/' + train.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)       # Read the image in grayscale
    image = preprocess(image)                               # Apply image preprocessing
    image = image / 255.0                                   # Normalize pixel values to [0, 1]
    train_x.append(image)
    
    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1} images")

print("Processing complete.")

valid_x = []

for i in range(valid_size):                                  # Loop through the validation dataset
    img_dir = '/kaggle/input/handwriting-recognition/validation_v2/validation/' + valid.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)        # Read the image in grayscale
    image = preprocess(image)                                # Apply image preprocessing
    image = image / 255.0                                    # Normalize pixel values to [0, 1]
    valid_x.append(image)

    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1} validation images")

print("Validation processing complete.")

# Convert 'train_x' to a NumPy array and reshape it
train_x = np.array(train_x).reshape(-1, 256, 64, 1)

# Convert 'valid_x' to a NumPy array and reshape it
valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)

alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "  # The set of valid characters
max_str_len = 24                             # Maximum length of input labels
num_of_characters = len(alphabets) + 1       # Number of unique characters, plus 1 for CTC pseudo-blank
num_of_timestamps = 64                       # Maximum length of predicted labels

def label_to_num(label):
    label_num = []                            # Initialize an empty list to store character indices
    for ch in label:                          # Iterate over each character in the input label
        label_num.append(alphabets.find(ch))  # Find the index of the character in the 'alphabets' string
        
    return np.array(label_num)                # Convert the list to a NumPy array and return it

def num_to_label(num):
    ret = ""                                  # Initialize an empty string to store the label
    for ch in num:                             # Iterate over each character index in the input list
        if ch == -1:                           # Check for the CTC Blank (a special case)
            break
        else:
            ret += alphabets[ch]               # Append the character from 'alphabets' based on the index
    return ret

name = 'UMANG'
print(name, '\n',label_to_num(name))

blank_label = -1

# Initialize arrays
train_y = np.ones([train_size, max_str_len]) * blank_label
train_label_len = np.zeros([train_size, 1])
train_input_len = np.ones([train_size, 1]) * (num_of_timestamps - 2)
train_output = np.zeros([train_size])

# Loop through the training data
for i in range(train_size):
    # Get the 'IDENTITY' value and convert it to a string if it's not already
    identity = train.loc[i, 'IDENTITY']
    if not isinstance(identity, str):
        identity = str(identity)
    
    # Set the label length and label values in train_y
    train_label_len[i] = len(identity)
    
    # Assuming label_to_num is a function that converts characters to numerical values
    train_y[i, 0:len(identity)] = label_to_num(identity)

# Initialize arrays for validation data
valid_y = np.ones([valid_size, max_str_len]) * blank_label
valid_label_len = np.zeros([valid_size, 1])
valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps - 2)
valid_output = np.zeros([valid_size])

# Loop through the validation data
for i in range(valid_size):
    # Get the 'IDENTITY' value and convert it to a string if it's not already
    identity = valid.loc[i, 'IDENTITY']
    if not isinstance(identity, str):
        identity = str(identity)
    
    # Set the label length and label values in valid_y
    valid_label_len[i] = len(identity)
    
    # Assuming label_to_num is a function that converts characters to numerical values
    valid_y[i, 0:len(identity)] = label_to_num(identity)

print('True label : ',train.loc[4, 'IDENTITY'] , '\ntrain_y : ',train_y[4],'\ntrain_label_len : ',train_label_len[4], 
      '\ntrain_input_len : ', train_input_len[4])

# Define the input layer with a shape of (256, 64, 1) for grayscale images
input_data = Input(shape=(256, 64, 1), name='input')

# Convolutional Layer 1: 32 filters, (3, 3) kernel, 'same' padding, He normal initialization
inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)
inner = BatchNormalization()(inner)                         # Batch normalization
inner = Activation('relu')(inner)                           # ReLU activation
inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # Max-pooling

# Convolutional Layer 2: 64 filters, (3, 3) kernel, 'same' padding, He normal initialization
inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)  
inner = Activation('relu')(inner)     
inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  
inner = Dropout(0.3)(inner)                                 

# Convolutional Layer 3: 128 filters, (3, 3) kernel, 'same' padding, He normal initialization
inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)                         
inner = Activation('relu')(inner)                           
inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  
inner = Dropout(0.3)(inner)                                 

# Reshape the output for sequence processing
inner = Reshape(target_shape=((64, 1024)), name='reshape')(inner)

# Fully Connected Layer 1: 64 units, ReLU activation, He normal initialization
inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

# Bidirectional LSTM Layers: 256 units, return sequences
inner = Bidirectional(LSTM(256, return_sequences=True), name='lstm1')(inner)
inner = Bidirectional(LSTM(256, return_sequences=True), name='lstm2')(inner)

# Output Layer: Number of characters, He normal initialization
inner = Dense(num_of_characters, kernel_initializer='he_normal', name='dense2')(inner)
y_pred = Activation('softmax', name='softmax')(inner)  # Softmax activation

# Create the model with input and output layers
model = Model(inputs=input_data, outputs=y_pred)

# Display a summary of the model architecture
model.summary()

# The ctc loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # The 2 is critical here since the first couple outputs of the RNN tend to be garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# Define input placeholders for true labels, input sequence length, and label sequence length
labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

# Calculate CTC loss using the ctc_lambda_func function
ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

# Create the final model that takes input data, true labels, input length, and label length
model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)

# Compile the final model with a dummy loss lambda function (loss calculation occurs elsewhere)
# The optimizer used is Adam with a learning rate of 0.0001
model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr=0.0001))

# Fit the model using training and validation data
model_final.fit(
    x=[train_x, train_y, train_input_len, train_label_len],                                # Input data and placeholders
    y=train_output,                                                                        # True output labels
    validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),  # Validation data
    epochs=60,                                                                             # Number of training epochs
    batch_size=128                                                                         # Batch size for training
)

preds = model.predict(valid_x)
decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], 
                                   greedy=True)[0][0])

prediction = []
for i in range(valid_size):
    prediction.append(num_to_label(decoded[i]))

y_true = valid.loc[0:valid_size, 'IDENTITY']
correct_char = 0
total_char = 0
correct = 0

for i in range(valid_size):
    pr = prediction[i]
    tr = y_true[i]
    total_char += len(tr)
    
    for j in range(min(len(tr), len(pr))):
        if tr[j] == pr[j]:
            correct_char += 1
            
    if pr == tr :
        correct += 1 
    
print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
print('Correct words predicted      : %.2f%%' %(correct*100/valid_size))

test = pd.read_csv('/kaggle/input/handwriting-recognition/written_name_test_v2.csv')

plt.figure(figsize=(15, 10))
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    img_dir = '/kaggle/input/handwriting-recognition/test_v2/test/'+test.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    
    image = preprocess(image)
    image = image/255.
    pred = model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                       greedy=True)[0][0])
    plt.title(num_to_label(decoded[0]), fontsize=12)
    plt.axis('off')
    
plt.subplots_adjust(wspace=0.2, hspace=-0.8)

# Initialize variables to track correct characters
correct_characters = 0
total_characters = 0
max_images = 100  # Specify the number of images to evaluate

# Iterate through the test set
for i in range(min(len(test), max_images)):
    # Get the ground truth label
    ground_truth = test.loc[i, 'IDENTITY']
    
    # Preprocess and predict the image using your model
    img_dir = '/kaggle/input/handwriting-recognition/test_v2/test/' + test.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    image = image / 255.0
    pred = model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], greedy=True)[0][0])
    predicted_label = num_to_label(decoded[0])

    # Calculate character-level accuracy
    for j in range(min(len(ground_truth), len(predicted_label))):
        if ground_truth[j] == predicted_label[j]:
            correct_characters += 1
        total_characters += 1

# Calculate character-level accuracy
character_level_accuracy = (correct_characters / total_characters) * 100

print('Character-Level Accuracy for {} images: {:.2f}%'.format(max_images, character_level_accuracy))

def predict(index):
    img_dir = '/kaggle/input/handwriting-recognition/test_v2/test/' + test.loc[index, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')

    image = preprocess(image)
    image = image / 255.

    # Reshape the image to have a batch size of 1
    image = image.reshape(1, 256, 64, 1)

    pred = model.predict(image)
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0])

    plt.title(num_to_label(decoded[0]), fontsize=12)
    plt.axis('off')
    plt.show()

predict(10261)