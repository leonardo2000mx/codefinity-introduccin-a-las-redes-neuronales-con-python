from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
# Ignore warnings
warnings.filterwarnings("ignore")
import os
os.system('wget https://codefinity-content-media.s3.eu-west-1.amazonaws.com/f9fc718f-c98b-470d-ba78-d84ef16ba45f/section_2/data.py 2>/dev/null')
from data import X_train, y_train, X_test, y_test

np.random.seed(1)

# Setup a random distribution of hyperparameters
param_distributions = {
    # 1. Specify layer sizes
    'hidden_layer_sizes': [(20, 20), (25, 25), (30, 30)],
    # 2. Specify learning rates
    'learning_rate_init': [0.02, 0.01, 0.005],
    # 3. Add parameter name for maximal number of iterations
    'max_iter': [10, 30, 50]
}

# Create the model
mlp = MLPClassifier()

# 4. Apply randomized search with 4 iterations and accuracy as the evaluation metric
random_search = RandomizedSearchCV(
    estimator=mlp,
    param_distributions=param_distributions,
    n_iter=4,
    scoring='accuracy',
    random_state=1
)
random_search.fit(X_train, y_train)

# Display the best parameters
print(f'Best parameters found: {random_search.best_params_}')

# Train the best model on the whole training data
best_mlp = random_search.best_estimator_
best_mlp.fit(X_train, y_train)

train_accuracy = accuracy_score(y_train, best_mlp.predict(X_train))
test_accuracy = accuracy_score(y_test, best_mlp.predict(X_test))

print(f'Train accuracy: {train_accuracy:.3f}')
print(f'Test accuracy: {test_accuracy:.3f}')