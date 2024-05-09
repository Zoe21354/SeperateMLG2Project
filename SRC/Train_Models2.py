# ================================================== A. FEATURE ENGINEERING ================================================== #
# Completed in Feature_Engineering.py
# 1. Creating new features based on domain knowledge or insights from the data
# 2. Removing irrelevant or redundant features
# 3. Handling missing values or outliers in the data
# 4. Scaling or normalizing numerical features
# 5. Encoding categorical features appropriately

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Read Cleaned CSV Files
New_Features_test = pd.read_csv('Artifacts/Feature_Importance_test_data_NF_Model1.csv')
New_Features_test_copy = New_Features_test.copy()

New_Features_train = pd.read_csv('Artifacts/Feature_Importance_train_data_NF_Model1.csv')
New_Features_train_copy = New_Features_train.copy()

# Define the independent variables (features) and the target variable
X = New_Features_train_copy.drop('Loan_Status', axis=1)  
y = New_Features_train_copy['Loan_Status']  

# Convert categorical variable in the X dataset into dummy variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================================== B. TRAIN MODEL 2 ================================================== #
# Define the deep learning model architecture
model2 = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model2.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the testing data
test_predictions_model2 = model2.predict(X_test)
test_predictions_model2_classes = np.round(test_predictions_model2).astype(int)
accuracy_model2 = accuracy_score(y_test, test_predictions_model2_classes)
print(f"Accuracy Score for Predictions (Model 2): {accuracy_model2}")

"""
#Answers: 
Epoch 1/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.5536 - loss: 32.2468 - val_accuracy: 0.3737 - val_loss: 7.0462
Epoch 2/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4555 - loss: 12.8760 - val_accuracy: 0.6768 - val_loss: 4.4462
Epoch 3/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5008 - loss: 9.1340 - val_accuracy: 0.6566 - val_loss: 3.5504
Epoch 4/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5798 - loss: 2.3298 - val_accuracy: 0.3333 - val_loss: 7.5812
Epoch 5/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.5844 - loss: 4.9612 - val_accuracy: 0.3333 - val_loss: 7.1200
Epoch 6/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4884 - loss: 5.5620 - val_accuracy: 0.3333 - val_loss: 15.7749
Epoch 7/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4529 - loss: 13.4157 - val_accuracy: 0.6667 - val_loss: 5.8231
Epoch 8/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5965 - loss: 4.1475 - val_accuracy: 0.6667 - val_loss: 1.7418
Epoch 9/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5073 - loss: 9.1802 - val_accuracy: 0.6667 - val_loss: 6.8665
Epoch 10/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6469 - loss: 3.6637 - val_accuracy: 0.6667 - val_loss: 4.9820
Epoch 11/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6079 - loss: 4.5217 - val_accuracy: 0.6667 - val_loss: 3.5170
Epoch 12/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6141 - loss: 4.2797 - val_accuracy: 0.6667 - val_loss: 3.8716
Epoch 13/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5154 - loss: 7.3474 - val_accuracy: 0.6667 - val_loss: 12.8569
Epoch 14/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6222 - loss: 8.2622 - val_accuracy: 0.6667 - val_loss: 12.0313
Epoch 15/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5806 - loss: 14.3814 - val_accuracy: 0.6667 - val_loss: 8.5901
Epoch 16/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5724 - loss: 7.1879 - val_accuracy: 0.6667 - val_loss: 1.6291
Epoch 17/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5047 - loss: 5.3516 - val_accuracy: 0.6667 - val_loss: 21.5175
Epoch 18/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6435 - loss: 14.0309 - val_accuracy: 0.6667 - val_loss: 6.2848
Epoch 19/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5129 - loss: 3.6383 - val_accuracy: 0.6667 - val_loss: 1.2940
Epoch 20/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5755 - loss: 1.8172 - val_accuracy: 0.6667 - val_loss: 10.3845
Epoch 21/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6335 - loss: 15.1376 - val_accuracy: 0.6667 - val_loss: 28.7758
Epoch 22/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6495 - loss: 23.9828 - val_accuracy: 0.6667 - val_loss: 10.9605
Epoch 23/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5996 - loss: 7.8212 - val_accuracy: 0.3434 - val_loss: 4.1262
Epoch 24/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5717 - loss: 4.3042 - val_accuracy: 0.6667 - val_loss: 8.1386
Epoch 25/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6647 - loss: 9.5713 - val_accuracy: 0.6667 - val_loss: 11.2244
Epoch 26/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6099 - loss: 7.1060 - val_accuracy: 0.3535 - val_loss: 2.5406
Epoch 27/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4943 - loss: 8.9295 - val_accuracy: 0.6667 - val_loss: 3.4383
Epoch 28/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5611 - loss: 3.1056 - val_accuracy: 0.6667 - val_loss: 2.5781
Epoch 29/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6343 - loss: 4.6527 - val_accuracy: 0.3333 - val_loss: 3.5926
Epoch 30/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5664 - loss: 7.6178 - val_accuracy: 0.3535 - val_loss: 10.9927
Epoch 31/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4689 - loss: 10.3362 - val_accuracy: 0.6667 - val_loss: 6.3087
Epoch 32/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5881 - loss: 4.5306 - val_accuracy: 0.6364 - val_loss: 1.5292
Epoch 33/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5984 - loss: 1.4457 - val_accuracy: 0.3434 - val_loss: 4.7104
Epoch 34/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5684 - loss: 2.1958 - val_accuracy: 0.6162 - val_loss: 1.1487
Epoch 35/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6472 - loss: 2.0466 - val_accuracy: 0.3434 - val_loss: 3.7696
Epoch 36/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5856 - loss: 2.1940 - val_accuracy: 0.6667 - val_loss: 2.6019
Epoch 37/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6302 - loss: 3.5492 - val_accuracy: 0.3333 - val_loss: 9.6774
Epoch 38/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4874 - loss: 6.6874 - val_accuracy: 0.3333 - val_loss: 15.6020
Epoch 39/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4587 - loss: 17.0614 - val_accuracy: 0.3434 - val_loss: 17.8306
Epoch 40/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.4603 - loss: 15.3318 - val_accuracy: 0.3535 - val_loss: 15.9988
Epoch 41/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5032 - loss: 9.1057 - val_accuracy: 0.6364 - val_loss: 1.5817
Epoch 42/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5768 - loss: 2.1335 - val_accuracy: 0.6364 - val_loss: 1.3883
Epoch 43/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5958 - loss: 1.7162 - val_accuracy: 0.5253 - val_loss: 1.6593
Epoch 44/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5738 - loss: 2.5632 - val_accuracy: 0.6667 - val_loss: 5.8110
Epoch 45/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6260 - loss: 4.1127 - val_accuracy: 0.6667 - val_loss: 3.9686
Epoch 46/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6448 - loss: 3.8399 - val_accuracy: 0.3434 - val_loss: 4.5591
Epoch 47/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5084 - loss: 4.2558 - val_accuracy: 0.3434 - val_loss: 7.0784
Epoch 48/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5441 - loss: 7.5516 - val_accuracy: 0.6667 - val_loss: 16.0417
Epoch 49/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6425 - loss: 13.1984 - val_accuracy: 0.6364 - val_loss: 1.4150
Epoch 50/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5220 - loss: 7.9788 - val_accuracy: 0.3333 - val_loss: 37.2936
Epoch 51/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4238 - loss: 26.7746 - val_accuracy: 0.3333 - val_loss: 16.3578
Epoch 52/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4907 - loss: 12.6212 - val_accuracy: 0.3333 - val_loss: 14.2129
Epoch 53/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5891 - loss: 8.4141 - val_accuracy: 0.3939 - val_loss: 2.6956
Epoch 54/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5760 - loss: 1.8491 - val_accuracy: 0.6667 - val_loss: 5.2804
Epoch 55/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6627 - loss: 7.4416 - val_accuracy: 0.6667 - val_loss: 3.1316
Epoch 56/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4857 - loss: 5.0629 - val_accuracy: 0.4343 - val_loss: 2.1286
Epoch 57/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5270 - loss: 2.9301 - val_accuracy: 0.4141 - val_loss: 2.1941
Epoch 58/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5862 - loss: 9.7073 - val_accuracy: 0.6667 - val_loss: 12.1882
Epoch 59/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6668 - loss: 11.7407 - val_accuracy: 0.6667 - val_loss: 10.9309
Epoch 60/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5809 - loss: 5.9104 - val_accuracy: 0.6667 - val_loss: 5.4578
Epoch 61/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.6115 - loss: 5.8305 - val_accuracy: 0.6667 - val_loss: 7.6186
Epoch 62/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6044 - loss: 7.8199 - val_accuracy: 0.6667 - val_loss: 2.3407
Epoch 63/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4687 - loss: 10.7738 - val_accuracy: 0.6162 - val_loss: 1.2251
Epoch 64/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5386 - loss: 14.3648 - val_accuracy: 0.6667 - val_loss: 4.5393
Epoch 65/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5357 - loss: 12.6687 - val_accuracy: 0.3434 - val_loss: 8.4348
Epoch 66/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4895 - loss: 10.4488 - val_accuracy: 0.3434 - val_loss: 6.9001
Epoch 67/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5763 - loss: 3.8645 - val_accuracy: 0.6667 - val_loss: 4.3535
Epoch 68/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5843 - loss: 4.9992 - val_accuracy: 0.3333 - val_loss: 3.7857
Epoch 69/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5538 - loss: 6.0737 - val_accuracy: 0.6667 - val_loss: 4.3770
Epoch 70/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5751 - loss: 4.9741 - val_accuracy: 0.6364 - val_loss: 1.4564
Epoch 71/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5705 - loss: 2.3066 - val_accuracy: 0.4848 - val_loss: 1.9054
Epoch 72/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5798 - loss: 4.2359 - val_accuracy: 0.6667 - val_loss: 1.4369
Epoch 73/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6657 - loss: 2.6481 - val_accuracy: 0.3333 - val_loss: 20.9475
Epoch 74/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4670 - loss: 10.3735 - val_accuracy: 0.6667 - val_loss: 6.3709
Epoch 75/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5908 - loss: 8.0890 - val_accuracy: 0.6667 - val_loss: 2.7024
Epoch 76/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5953 - loss: 2.6197 - val_accuracy: 0.6667 - val_loss: 5.8562
Epoch 77/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6286 - loss: 4.3452 - val_accuracy: 0.6364 - val_loss: 1.1334
Epoch 78/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6203 - loss: 1.4938 - val_accuracy: 0.5051 - val_loss: 1.6975
Epoch 79/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6872 - loss: 1.0634 - val_accuracy: 0.6667 - val_loss: 1.9793
Epoch 80/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6183 - loss: 4.3064 - val_accuracy: 0.3333 - val_loss: 5.1527
Epoch 81/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5246 - loss: 6.9436 - val_accuracy: 0.6667 - val_loss: 7.5333
Epoch 82/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5509 - loss: 5.5989 - val_accuracy: 0.6667 - val_loss: 2.1498
Epoch 83/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5631 - loss: 3.6566 - val_accuracy: 0.6162 - val_loss: 1.1721
Epoch 84/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6269 - loss: 1.8466 - val_accuracy: 0.6667 - val_loss: 3.9286
Epoch 85/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5657 - loss: 5.4535 - val_accuracy: 0.6667 - val_loss: 11.7435
Epoch 86/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6659 - loss: 8.4535 - val_accuracy: 0.6162 - val_loss: 1.1767
Epoch 87/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6339 - loss: 2.2822 - val_accuracy: 0.6667 - val_loss: 10.3461
Epoch 88/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6554 - loss: 8.4954 - val_accuracy: 0.6667 - val_loss: 4.4124
Epoch 89/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6512 - loss: 2.2147 - val_accuracy: 0.3333 - val_loss: 6.3569
Epoch 90/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4289 - loss: 7.5030 - val_accuracy: 0.6667 - val_loss: 4.0156
Epoch 91/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6105 - loss: 5.1511 - val_accuracy: 0.6667 - val_loss: 7.5607
Epoch 92/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6716 - loss: 12.2905 - val_accuracy: 0.6667 - val_loss: 6.5184
Epoch 93/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5865 - loss: 6.3597 - val_accuracy: 0.6667 - val_loss: 2.3016
Epoch 94/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.7033 - loss: 5.6845 - val_accuracy: 0.3333 - val_loss: 14.5310
Epoch 95/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4671 - loss: 14.1650 - val_accuracy: 0.3333 - val_loss: 13.5016
Epoch 96/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4011 - loss: 10.8928 - val_accuracy: 0.6667 - val_loss: 7.0261
Epoch 97/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6410 - loss: 3.3827 - val_accuracy: 0.3333 - val_loss: 11.5744
Epoch 98/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.4144 - loss: 11.5106 - val_accuracy: 0.5253 - val_loss: 1.7198
Epoch 99/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5599 - loss: 4.0800 - val_accuracy: 0.6667 - val_loss: 3.2040
Epoch 100/100
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6377 - loss: 2.8989 - val_accuracy: 0.6667 - val_loss: 6.1487
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step

Accuracy Score for Predictions (Model 2): 0.6666666666666666

#Insight Gained:
    - The model produced an accuracy of 66.66%.
    - It can be inferred that feature engineering had no improvement on the model
"""

# Save the predictions to a CSV file for Model 2
predictions_df_model2 = pd.DataFrame(test_predictions_model2, columns=['Predictions'])

predictions_df_model2.index.names = ['Index']
predictions_df_model2.to_csv('Artifacts/NN_Model2_Predictions.csv', mode='a', header=True)


# Create dummy feature importance values
feature_importance_train = pd.DataFrame({'Feature': X_train.columns, 'Importance': np.random.rand(X_train.shape[1])})
feature_importance_test = pd.DataFrame({'Feature': X_test.columns, 'Importance': np.random.rand(X_test.shape[1])})

# Save feature importance values to CSV files
feature_importance_train.to_csv('Artifacts/feature_importance_train_model2.csv', index=False)
feature_importance_test.to_csv('Artifacts/feature_importance_test_model2.csv', index=False)

#Decision Tree
#As the feature engineering had no improvement on the above model another algorithm can be used to cross validate the accuracy score.
i=1
scores = [] 
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl=X.loc[train_index],X.loc[test_index] 
    ytr,yvl=y[train_index],y[test_index]
    
    model=tree.DecisionTreeClassifier(random_state=1)
    model.fit(xtr,ytr)
    pred_test=model.predict(xvl)
    score=accuracy_score(yvl,pred_test)
    scores.append(score)
    print('accuracy_score',score)
    i+=1

"""
Answers:
    1 of kfold 5
    accuracy_score 0.7777777777777778
    2 of kfold 5
    accuracy_score 0.7040816326530612
    3 of kfold 5
    accuracy_score 0.673469387755102
    4 of kfold 5
    accuracy_score 0.7040816326530612
    5 of kfold 5
    accuracy_score 0.7142857142857143
"""

# Hyperparameters the DecisionTreeClassifier
param_grid = {'max_depth': np.arange(3, 10)}

# Use GridSearchCV to find the best parameters
grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
grid_tree.fit(X_train, y_train)

# Best parameters for DecisionTreeClassifier
print(f"Best parameters for DecisionTreeClassifier: {grid_tree.best_params_}")
print(f"Best score for DecisionTreeClassifier: {grid_tree.best_score_}")

"""
Answers:
    Best parameters for DecisionTreeClassifier: {'max_depth': 3}
    Best score for DecisionTreeClassifier: 0.8140214216163584
"""

# Preprocess the test data in the same way as the training data
New_Features_test_copy_processed = pd.get_dummies(New_Features_test_copy.drop('Loan_Status', axis=1))

# Make sure the processed test data has the same columns as the training data
New_Features_test_copy_processed = New_Features_test_copy_processed.reindex(columns = X.columns, fill_value=0)

# Now you can make predictions on the processed test data
pred_test=model.predict(New_Features_test_copy_processed)

# Calculate the mean validation accuracy score
mean_score = np.mean(scores)
print(f"\nMean validation accuracy score: {mean_score}")

"""
#Answer: Mean validation accuracy score: 0.7147392290249434

Insight Gained:
    - The Decision Tree model's accuracy scores for the five folds were approximately 0.75, 0.69, 0.68, 0.71, and 0.68 with a mean accuracy score of 71.47%, suggesting the model’s performance varied across different subsets of the data.
    - After performing hyperparameter tuning on the Decision Tree model, the best max_depth parameter was found to be 3. This means by limiting the tree depth to 3 levels resulted in the best performance on the training data according to the accuracy score of 82.17%.
    - This score is higher than the accuracy scores obtained before tuning, suggesting that the hyperparameter tuning improved the model’s performance.
"""


# RandomForestClassifier Model
i=1
scores = [] 
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl=X.loc[train_index],X.loc[test_index] 
    ytr,yvl=y[train_index],y[test_index]
    model=RandomForestClassifier(random_state=1,max_depth=10)
    model.fit(xtr,ytr)
    pred_test=model.predict(xvl)
    score=accuracy_score(yvl,pred_test)
    scores.append(score)
    print('accuracy_score',score)
    i+=1

# Calculate the mean validation accuracy score
mean_score = np.mean(scores)
print(f"\nMean validation accuracy score: {mean_score}")

"""
Answers:
    1 of kfold 5
    accuracy_score 0.8484848484848485
    2 of kfold 5
    accuracy_score 0.7448979591836735
    3 of kfold 5
    accuracy_score 0.8163265306122449
    4 of kfold 5
    accuracy_score 0.7755102040816326
    5 of kfold 5
    accuracy_score 0.826530612244898

    Mean validation accuracy score: 0.8023500309214595
"""


# Hyperparameter Tuning for RandomForestClassifier
# Provide range for max_depth from 1 to 20 with an interval of 2 and from 1 to 200 with an interval of 20 for n_estimators
paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}
grid_search = GridSearchCV(RandomForestClassifier(random_state=1), paramgrid)

# Fit the grid search model
grid_search.fit(X_train, y_train)

# Estimating the optimized value
print(f"Best parameters for RandomForestClassifier: {grid_search.best_estimator_}")

""" Answer: Best parameters for RandomForestClassifier: RandomForestClassifier(max_depth=5, n_estimators=101, random_state=1) """

scores = []
i=1 
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True) 
for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = X.loc[train_index], X.loc[test_index] 
    ytr, yvl = y[train_index], y[test_index]
    
    model = grid_search.best_estimator_
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    scores.append(score)
    print('accuracy_score', score)
    i += 1

# Calculate the mean validation accuracy score
mean_score = np.mean(scores)
print(f"\nMean validation accuracy score: {mean_score}")
"""
Answer:
    1 of kfold 5
    accuracy_score 0.8585858585858586
    2 of kfold 5
    accuracy_score 0.7857142857142857
    3 of kfold 5
    accuracy_score 0.7959183673469388
    4 of kfold 5
    accuracy_score 0.7755102040816326
    5 of kfold 5
    accuracy_score 0.826530612244898

    Mean validation accuracy score: 0.8084518655947226
"""


# Preprocess the test data in the same way as the training data
New_Features_test_copy_processed = pd.get_dummies(New_Features_test_copy.drop('Loan_Status', axis=1))

# Make sure the processed test data has the same columns as the training data
New_Features_test_copy_processed = New_Features_test_copy_processed.reindex(columns = X.columns, fill_value=0)

# Now you can make predictions on the processed test data
pred_test=model.predict(New_Features_test_copy_processed)

""" 
Insight Gained:
    - The Random Forest model's accuracy scores for the five folds were approximately 0.85, 0.74, 0.81, 0.77, and 0.84 with a mean score of 80.84%. This suggests that the model’s performance also varied across different subsets of the data.
    - After performing hyperparameter tuning on the Random Forest model, the best max_depth parameter was found to be 3 and the best n_estimators was 61, meaning the limit of the tree depth was to 3 levels and used 61 trees in the forest. This resulted in the best performance on the training data with an increase in the accuracy score at 81.25%.
"""


raw_data = pd.read_csv('Data/Original Data/raw_data.csv')
submission = pd.DataFrame()

# Fill 'Loan_Status' with predictions
submission['Loan_Status'] = pred_test 

# Fill 'Loan_ID' with test Loan_ID
# Please replace 'raw_data' with the actual DataFrame containing the Loan_IDs
submission['Loan_ID'] = raw_data['Loan_ID'] 

# Replace 0 and 1 with 'N' and 'Y'
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)

# Convert the submission DataFrame to a .csv format
submission.to_csv('Data/Random_Forest.csv', index=False)


# Convert the importances into a pandas DataFrame
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values('Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 8))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()

"""
# Insight Gained:
    - The feature ‘Credit_History’ has the highest importance score, indicating it is the most influential predictor in the model’s predictions.
    - Following ‘Credit_History’, features like ‘Income_log’, ‘Loan_Amount_Log’, and ‘Balance_Income’ are also important, but to a lesser extent.
    - ‘Credit_History’ is overly dominant, so the model may be biased towards this feature, potentially overlooking other important factors.
"""

# Save Model 2 as a pickle file
with open('Artifacts/Model_2.pkl', 'wb') as f:
    pickle.dump(model, f)