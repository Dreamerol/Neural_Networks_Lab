import tensorflow as tf
import numpy as np
#model to predict the vector in which quadrant is
X = np.array([[1, 2], [1, 5], [-2,3], [-3, 2], [4, -5], [5, -12], [-6, -3], [-7, 10], [8, -4], [2, -9]])
y = np.array([[1], [1], [2], [2], [4], [4], [3], [2], [4], [4]])

x_cords = np.linspace(-10, 10, 10)
y_cords = np.linspace(-10, 10, 10)
points = [[float(x),float(y)] for x in x_cords for y in y_cords]
print(points)

training_points = np.array(points[:len(points)//2])
testing_points = np.array(points[len(points)//2:len(points)])

def giveQuadrant(point):
  if point[0] > 0 and point[1] > 0:
    return [1,0,0,0]
  elif point[0] < 0 and point[1] > 0:
    return [0,1,0,0]
  elif point[0] < 0 and point[1] < 0:
    return [0,0,1,0]
  else:
    return [0,0,0,1]

output_training = np.array([giveQuadrant(x) for x in training_points])
output_testing = np.array([giveQuadrant(x) for x in testing_points])
print(output_training)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(4, activation='softmax')
    ])
#the number of neurons in the last layer is equal to the number of categories

model.compile(
    optimizer='adam',
    loss='CategoricalCrossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=100, verbose=0)

test_data = np.array([[3, 2], [-1, -3], [2,3], [3, 12], [-4, 5], [2, -1], [4, -3], [7, 1], [3, -4], [2, 2]])
output_data = np.array([[1], [3], [1], [1], [2], [4], [4], [1], [4], [1]])

predictions = model.predict(test_data)

for num, pred,out in zip(test_data, predictions, output_data):
  print(np.argmax(pred) + 1, out)
  if np.argmax(pred) + 1 == out:
    print("correct\n")
  else:
    print("correct\n")

model.fit(training_points, output_training, epochs=100, verbose=0)
predictionsMorePoints = model.predict(testing_points)

for pred, real in zip(predictionsMorePoints, output_testing):
  print(pred, real)
  if real[np.argmax(pred)] == 1:
    print("correct\n")
  else:
    print("incorrect\n")
