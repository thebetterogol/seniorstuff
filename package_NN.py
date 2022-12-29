from sklearn.neural_network import MLPClassifier
import random
import numpy as np
import pandas as pd

#breast cancer data

df = pd.read_csv("breast_cancer_data.csv")
df_means = df.iloc[:, range(0,12)].copy()
df_se = df.iloc[:, [0,1] + list(range(12,22))]
df_worst = df.iloc[:, [0,1] + list(range(23,32))]

# replace B with 0 and M with 1

df_means.loc[list(df_means['diagnosis'] == 'B'), 'diagnosis'] = 0
df_means.loc[list(df_means['diagnosis'] == 'M'), 'diagnosis'] = 1

#split dataset. 80% training and 20% testing

indices = list(range(0, len(df_means)))
random.shuffle(indices)

train_indices = indices[0:int(len(indices)*0.8)]

test_indices = list(set(indices) - set(train_indices))

training = df_means.iloc[train_indices]
testing = df_means.iloc[test_indices]

# training data
x_train = np.array(training.iloc[:, 2:].values.tolist())
y_train = np.array(list(map(lambda el:[el], training.iloc[:, 1].values))).ravel()

# clf = MLPClassifier(activation='logistic', solver='sgd', alpha=1e-5, learning_rate='constant', learning_rate_init=0.1, 
# hidden_layer_sizes=(6), random_state=1)
clf = MLPClassifier(activation='logistic', solver='adam', learning_rate='constant',
learning_rate_init=0.001, hidden_layer_sizes=(18, 2), random_state=1, max_iter=1000)

clf.fit(x_train, y_train)

x_test = np.array(testing.iloc[:, 2:].values.tolist())
y_test = np.array(list(map(lambda el:[el], testing.iloc[:, 1].values))).ravel()

predictions = clf.predict(x_test)

#find number of same values
num_same = 0
num_diff = 0
for j in range(0, len(predictions)):
    if predictions[j] == y_test[j]:
        num_same += 1
    else:
        num_diff += 1

print(predictions)

print(num_same, num_diff)