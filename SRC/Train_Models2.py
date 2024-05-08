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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

# Read Cleaned CSV Files
New_Features_test = pd.read_csv('Artifacts/Feature_Importance_test_data_NF_Model1.csv')
New_Features_test_copy = New_Features_test.copy()

New_Features_train = pd.read_csv('Artifacts/Feature_Importance_train_data_NF_Model.csv')
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
4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 44ms/step - accuracy: 0.3825 - loss: 288.2667 - val_accuracy: 0.6400 - val_loss: 93.3430
Epoch 2/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6279 - loss: 83.7408 - val_accuracy: 0.6400 - val_loss: 125.3931
Epoch 3/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6529 - loss: 84.8350 - val_accuracy: 0.6400 - val_loss: 50.6709
Epoch 4/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5553 - loss: 25.8646 - val_accuracy: 0.4000 - val_loss: 20.4735
Epoch 5/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.4594 - loss: 16.1814 - val_accuracy: 0.6400 - val_loss: 32.4772
Epoch 6/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6861 - loss: 14.6053 - val_accuracy: 0.3200 - val_loss: 13.1153
Epoch 7/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.4437 - loss: 12.8190 - val_accuracy: 0.6800 - val_loss: 21.3057
Epoch 8/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.4673 - loss: 14.0593 - val_accuracy: 0.6800 - val_loss: 16.9847
Epoch 9/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6819 - loss: 9.3870 - val_accuracy: 0.3600 - val_loss: 24.6760
Epoch 10/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.3241 - loss: 35.0079 - val_accuracy: 0.6400 - val_loss: 27.9458
Epoch 11/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5916 - loss: 22.0611 - val_accuracy: 0.6800 - val_loss: 19.8524
Epoch 12/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5552 - loss: 13.5718 - val_accuracy: 0.6000 - val_loss: 11.5013
Epoch 13/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5961 - loss: 12.1485 - val_accuracy: 0.6800 - val_loss: 20.7940
Epoch 14/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5241 - loss: 13.7204 - val_accuracy: 0.5600 - val_loss: 11.1837
Epoch 15/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6840 - loss: 8.4772 - val_accuracy: 0.6800 - val_loss: 22.3395
Epoch 16/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.5892 - loss: 11.9219 - val_accuracy: 0.3600 - val_loss: 24.9924
Epoch 17/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 0.3442 - loss: 28.2977 - val_accuracy: 0.6800 - val_loss: 24.3118
Epoch 18/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6364 - loss: 12.8042 - val_accuracy: 0.3200 - val_loss: 10.7558
Epoch 19/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.5430 - loss: 9.0528 - val_accuracy: 0.6800 - val_loss: 21.5486
Epoch 20/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6872 - loss: 12.2429 - val_accuracy: 0.3600 - val_loss: 11.5797
Epoch 21/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.3673 - loss: 12.2365 - val_accuracy: 0.6800 - val_loss: 19.5020
Epoch 22/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5794 - loss: 9.5985 - val_accuracy: 0.6800 - val_loss: 12.9773
Epoch 23/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.6893 - loss: 9.7151 - val_accuracy: 0.3600 - val_loss: 15.2136
Epoch 24/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.3367 - loss: 22.6544 - val_accuracy: 0.6400 - val_loss: 28.8194
Epoch 25/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6238 - loss: 21.4520 - val_accuracy: 0.6800 - val_loss: 17.6201
Epoch 26/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.4911 - loss: 12.4015 - val_accuracy: 0.5200 - val_loss: 9.6663
Epoch 27/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6643 - loss: 6.8481 - val_accuracy: 0.4000 - val_loss: 8.9305
Epoch 28/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.3831 - loss: 12.8510 - val_accuracy: 0.6400 - val_loss: 34.2713
Epoch 29/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6436 - loss: 27.0624 - val_accuracy: 0.6400 - val_loss: 40.6725
Epoch 30/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6747 - loss: 21.8212 - val_accuracy: 0.3600 - val_loss: 24.1713
Epoch 31/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.3666 - loss: 29.1916 - val_accuracy: 0.6400 - val_loss: 24.7024
Epoch 32/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6208 - loss: 14.0178 - val_accuracy: 0.3600 - val_loss: 8.7525
Epoch 33/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.5164 - loss: 5.7991 - val_accuracy: 0.6800 - val_loss: 14.8070
Epoch 34/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5768 - loss: 7.4518 - val_accuracy: 0.3600 - val_loss: 10.1482
Epoch 35/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.4367 - loss: 9.8162 - val_accuracy: 0.3600 - val_loss: 8.5103
Epoch 36/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.3979 - loss: 11.6476 - val_accuracy: 0.6400 - val_loss: 26.2485
Epoch 37/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6437 - loss: 16.7170 - val_accuracy: 0.4000 - val_loss: 7.6788
Epoch 38/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.3864 - loss: 8.9269 - val_accuracy: 0.6400 - val_loss: 25.8739
Epoch 39/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6124 - loss: 18.2781 - val_accuracy: 0.6800 - val_loss: 17.0770
Epoch 40/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5522 - loss: 10.4688 - val_accuracy: 0.6400 - val_loss: 43.6226
Epoch 41/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6560 - loss: 54.9425 - val_accuracy: 0.6400 - val_loss: 168.5073
Epoch 42/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6685 - loss: 131.9707 - val_accuracy: 0.6400 - val_loss: 196.8980
Epoch 43/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6779 - loss: 136.5417 - val_accuracy: 0.6400 - val_loss: 162.5853
Epoch 44/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6487 - loss: 112.1129 - val_accuracy: 0.6400 - val_loss: 77.8639
Epoch 45/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6291 - loss: 42.8160 - val_accuracy: 0.3600 - val_loss: 57.2899
Epoch 46/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.3721 - loss: 69.1717 - val_accuracy: 0.4800 - val_loss: 7.1171
Epoch 47/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5959 - loss: 12.3284 - val_accuracy: 0.6400 - val_loss: 53.6568
Epoch 48/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.7008 - loss: 30.7733 - val_accuracy: 0.6800 - val_loss: 10.4793
Epoch 49/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5802 - loss: 32.4824 - val_accuracy: 0.3600 - val_loss: 46.4382
Epoch 50/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.4977 - loss: 34.5550 - val_accuracy: 0.6400 - val_loss: 54.6070
Epoch 51/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6831 - loss: 43.7355 - val_accuracy: 0.6400 - val_loss: 69.1301
Epoch 52/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6581 - loss: 44.4245 - val_accuracy: 0.6400 - val_loss: 8.6961
Epoch 53/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.5303 - loss: 16.9109 - val_accuracy: 0.4800 - val_loss: 6.9051
Epoch 54/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6953 - loss: 9.7746 - val_accuracy: 0.6400 - val_loss: 52.9581
Epoch 55/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6081 - loss: 41.1198 - val_accuracy: 0.6800 - val_loss: 16.6141
Epoch 56/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.4653 - loss: 12.6590 - val_accuracy: 0.3600 - val_loss: 15.5821
Epoch 57/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.4437 - loss: 18.7781 - val_accuracy: 0.6400 - val_loss: 25.4163
Epoch 58/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6459 - loss: 12.0518 - val_accuracy: 0.3600 - val_loss: 18.3088
Epoch 59/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.4220 - loss: 21.1886 - val_accuracy: 0.6400 - val_loss: 24.1997
Epoch 60/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5734 - loss: 13.7745 - val_accuracy: 0.6400 - val_loss: 62.1522
Epoch 61/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6768 - loss: 72.9597 - val_accuracy: 0.6400 - val_loss: 217.8488
Epoch 62/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6393 - loss: 173.9278 - val_accuracy: 0.6400 - val_loss: 249.2299
Epoch 63/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6758 - loss: 179.5259 - val_accuracy: 0.6400 - val_loss: 196.0006
Epoch 64/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6373 - loss: 142.9431 - val_accuracy: 0.6400 - val_loss: 97.9429
Epoch 65/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6362 - loss: 58.3077 - val_accuracy: 0.3600 - val_loss: 21.1403
Epoch 66/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.3471 - loss: 47.3285 - val_accuracy: 0.3600 - val_loss: 49.4839
Epoch 67/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.3883 - loss: 31.9780 - val_accuracy: 0.6400 - val_loss: 37.6405
Epoch 68/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6341 - loss: 30.3804 - val_accuracy: 0.6400 - val_loss: 43.8953
Epoch 69/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6643 - loss: 26.5868 - val_accuracy: 0.4800 - val_loss: 7.3514
Epoch 70/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5099 - loss: 14.4995 - val_accuracy: 0.3600 - val_loss: 10.3971
Epoch 71/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.4417 - loss: 10.3357 - val_accuracy: 0.6400 - val_loss: 26.9642
Epoch 72/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6561 - loss: 14.8698 - val_accuracy: 0.5200 - val_loss: 7.7908
Epoch 73/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.4694 - loss: 7.9368 - val_accuracy: 0.6800 - val_loss: 16.0751
Epoch 74/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6394 - loss: 10.4436 - val_accuracy: 0.5200 - val_loss: 7.8412
Epoch 75/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.4908 - loss: 4.6867 - val_accuracy: 0.3600 - val_loss: 6.8942
Epoch 76/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.5131 - loss: 4.7735 - val_accuracy: 0.6800 - val_loss: 10.8589
Epoch 77/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6181 - loss: 5.4314 - val_accuracy: 0.6800 - val_loss: 15.6753
Epoch 78/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6613 - loss: 12.0182 - val_accuracy: 0.6800 - val_loss: 15.8348
Epoch 79/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6172 - loss: 6.9283 - val_accuracy: 0.3600 - val_loss: 8.3732
Epoch 80/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5627 - loss: 6.2230 - val_accuracy: 0.6400 - val_loss: 25.3491
Epoch 81/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6560 - loss: 15.2657 - val_accuracy: 0.4000 - val_loss: 6.5883
Epoch 82/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5496 - loss: 4.1680 - val_accuracy: 0.3600 - val_loss: 8.9747
Epoch 83/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.4448 - loss: 7.3670 - val_accuracy: 0.6800 - val_loss: 13.3440
Epoch 84/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5981 - loss: 5.8806 - val_accuracy: 0.3600 - val_loss: 11.6454
Epoch 85/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.3686 - loss: 13.5325 - val_accuracy: 0.6800 - val_loss: 15.5139
Epoch 86/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5272 - loss: 9.1142 - val_accuracy: 0.6800 - val_loss: 8.7325
Epoch 87/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.6550 - loss: 6.5467 - val_accuracy: 0.4000 - val_loss: 6.1632
Epoch 88/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.4378 - loss: 4.2680 - val_accuracy: 0.6800 - val_loss: 12.8487
Epoch 89/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.6393 - loss: 7.7373 - val_accuracy: 0.3600 - val_loss: 11.1603
Epoch 90/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.3449 - loss: 19.2652 - val_accuracy: 0.4000 - val_loss: 5.9291
Epoch 91/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.4998 - loss: 7.5575 - val_accuracy: 0.4000 - val_loss: 5.8403
Epoch 92/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.3139 - loss: 9.7641 - val_accuracy: 0.6800 - val_loss: 12.8780
Epoch 93/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6665 - loss: 7.0197 - val_accuracy: 0.6000 - val_loss: 6.9502
Epoch 94/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.6190 - loss: 3.4661 - val_accuracy: 0.6800 - val_loss: 9.7559
Epoch 95/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - accuracy: 0.6354 - loss: 4.4283 - val_accuracy: 0.3200 - val_loss: 5.6791
Epoch 96/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.4718 - loss: 3.8436 - val_accuracy: 0.6000 - val_loss: 6.6140
Epoch 97/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5026 - loss: 4.7151 - val_accuracy: 0.6800 - val_loss: 12.1752
Epoch 98/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5733 - loss: 6.7372 - val_accuracy: 0.5600 - val_loss: 6.0589
Epoch 99/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5906 - loss: 4.9796 - val_accuracy: 0.6000 - val_loss: 6.2254
Epoch 100/100
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.5541 - loss: 3.8409 - val_accuracy: 0.6800 - val_loss: 9.9745
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step

Accuracy Score for Predictions (Model 2): 0.68

#Insight Gained:
    - The model produced an accuracy of 68%.
    - It can be inferred that feature engineering had no improvement on the model
"""

# Save the predictions to a CSV file for Model 2
predictions_df_model2 = pd.DataFrame(test_predictions_model2, columns=['Predictions'])

predictions_df_model2.index.names = ['Index']
predictions_df_model2.to_csv('Artifacts/NN_Model2_Predictions.csv', mode='a', header=True)

# Save Model 2 as a pickle file
with open('Artifacts/Model_2.pkl', 'wb') as f:
    pickle.dump(model2, f)

# Create dummy feature importance values
feature_importance_train = pd.DataFrame({'Feature': X_train.columns, 'Importance': np.random.rand(X_train.shape[1])})
feature_importance_test = pd.DataFrame({'Feature': X_test.columns, 'Importance': np.random.rand(X_test.shape[1])})

# Save feature importance values to CSV files
feature_importance_train.to_csv('Artifacts/feature_importance_train_model2.csv', index=False)
feature_importance_test.to_csv('Artifacts/feature_importance_test_model2.csv', index=False)


