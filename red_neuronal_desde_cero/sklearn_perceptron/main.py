from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
# Ignore warnings
warnings.filterwarnings("ignore")
import os
os.system('wget https://codefinity-content-media.s3.eu-west-1.amazonaws.com/f9fc718f-c98b-470d-ba78-d84ef16ba45f/section_2/perceptron.py 2>/dev/null')
from perceptron import X, y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
# 1. Initialize a perceptron
model = MLPClassifier(max_iter=100, hidden_layer_sizes=(6, 6), learning_rate_init=0.01, random_state=10)
# 2. Train the model
model.fit(X_train, y_train)
# 3. Obtain predictions on the test set
y_pred = model.predict(X_test)
# 4. Compute the accuracy on the test set
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {score:.3f}')