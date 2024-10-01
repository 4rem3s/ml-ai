import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(42)
X_train = np.random.rand(100,1)
y_train = 3 * X_train + 2 + np.random.rand(100, 1) * 0.1

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
    ])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate= 0.01),
              loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=500, verbose=0)

X_test = np.array([[0], [1]])
y_pred = model.predict(X_test)

plt.scatter(X_train, y_train, label = 'training data')
plt.plot(X_test, y_pred, color='red', label='prediction')
plt.title('Linear regression using Nnet')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


weights, biases = model.layers[0].get_weights()
print(f"Learned weight : { weights[0][0] :.4f}, Learned bias : {biases[0]:.4f}")
