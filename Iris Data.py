
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd        
import tensorflow as tf
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn.metrics import classification_report


# In[2]:


# load iris dataset
iris = datasets.load_iris()

# extract data and target 
X = iris.data
y = pd.DataFrame(iris.target, columns=["Type"])

# puts data into data frame and combines 
df = pd.DataFrame(X, columns=iris.feature_names)
df = pd.concat([y, df], axis=1)
print(df.head(5))

# randomizes data
df = df.sample(frac=1).reset_index(drop=True)


# splits data into test and train
df_train, df_test = np.split(df, [int(.7*len(df))])

df_test = df_test.reset_index(drop=True)

print(df_train.info())
df_test.info()


# In[3]:


X_train = df_train.drop(['Type'], axis=1)
y_train = df_train['Type']


# In[4]:


X_train.head(5)


# In[5]:


X_train = X_train.values
X_test = df_test.drop(['Type'], axis=1).values
y_test = df_test['Type'].values

print(X_test)


# Gradient Boosting

# In[6]:


from sklearn.ensemble import GradientBoostingClassifier
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
n_estimators = [50, 100, 150, 200, 250]
for learning_rate in learning_rates:
    for n_estimator in n_estimators:
        gb = GradientBoostingClassifier(n_estimators=n_estimator, learning_rate=learning_rate, max_features=4, max_depth=7)
        gb.fit(X_train, y_train)
        #     gb_predict = pd.DataFrame(gb.predict(X_test), columns=['Gradient Boost Predictions'])
        print(n_estimator, " estimators, learning rate of ", learning_rate, ", accuracy = ", gb.score(X_test, y_test))


# Random Forest

# In[7]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, max_features=4)
random_forest.fit(X_train, y_train)
rand_for_predict = random_forest.predict(X_test)
print(rand_for_predict)
rand_for_predict = pd.DataFrame(rand_for_predict, columns=['Random Forest Predictions'])
print(random_forest.score(X_test, y_test))



# XGBoost

# In[8]:


from xgboost import XGBClassifier
xgb_model = xgboost.XGBClassifier(n_estimators=300, learning_rate=0.1, use_label_encoder=False)
xgb_model.fit(X_train, y_train)
xgb_predict = pd.DataFrame(xgb_model.predict(X_test), columns=['XGBoost Predictions'])

print(xgb_predict)
xgb_model.score(X_test, y_test)


# Neural Network

# In[9]:


def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs,:], y_data[idxs]


# In[10]:


epochs = 150
batch_size = 100
X_test = tf.Variable(X_test)


# In[11]:


w1 = tf.Variable(tf.random.normal([4, 128], stddev=0.03), name='w1')
b1 = tf.Variable(tf.random.normal([128]), name='b1')

w2 = tf.Variable(tf.random.normal([128, 10]), name='w2')
b2 = tf.Variable(tf.random.normal([10]), name='b2')


# In[12]:


def nn_model(x_input, w1, b1, w2, b2):
    x = tf.add(tf.matmul(tf.cast(x_input, tf.float32), w1), b1)
#     print(x)
    x = tf.nn.relu(x)
    logits = tf.add(tf.matmul(x, w2), b2)
    return logits


# In[13]:


def loss_fn(logits, labels):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return cross_entropy


# In[14]:


optimizer = tf.keras.optimizers.Adam()


# In[15]:


total_batch = int(len(y_train)/batch_size)
for epoch in range(epochs):
    avg_loss = 0
    nn_predict = list()
    for i in range(total_batch):
        batch_x, batch_y = get_batch(X_train, y_train, batch_size=batch_size)
        batch_x = tf.Variable(batch_x)
        batch_y = tf.Variable(batch_y)
        batch_y = tf.one_hot(batch_y, 10)
        with tf.GradientTape() as tape:
            logits = nn_model(batch_x, w1, b1, w2, b2)
            loss = loss_fn(logits, batch_y)
        gradients = tape.gradient(loss, [w1, b1, w2, b2])
        optimizer.apply_gradients(zip(gradients, [w1, b1, w2, b2]))
        avg_loss += loss/total_batch
        test_logits = nn_model(X_test, w1, b1, w2, b2)
    max_idxs = tf.argmax(test_logits, axis=1)
    test_acc = np.sum(max_idxs.numpy() == y_test) / len(y_test)
    print(f"Epoch: {epoch + 1}, loss={avg_loss:.3f}, test set accuracy={test_acc*100:.3f}%")
    nn_predict.append(max_idxs.numpy())
#     print(nn_predict)
nn_predict = pd.DataFrame(nn_predict, index=['Neural Network Predictions']).transpose()
print(nn_predict)


# In[16]:


output = pd.concat([df_test, rand_for_predict, xgb_predict, nn_predict], axis=1)
print(output.head(5))
# output.to_csv('iris_output.csv')


# In[17]:


print("Scores for Random Forest Model: \n"+ classification_report(output['Type'], output['Random Forest Predictions']))


# In[18]:


print("Scores for XGBoost Model: \n"+ classification_report(output['Type'], output['XGBoost Predictions']))


# In[19]:


print("Scores for Neural Network: \n"+ classification_report(output['Type'], output['Neural Network Predictions']))

