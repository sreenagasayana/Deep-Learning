import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Creating Session
session = tf.Session()

# Creating Constant Matrix
matrixA = tf.constant([2, 4, 6, 8], shape = [2,2])

print("Matrix A : \n", session.run(matrixA))

# Creating Constant Matrix
matrixB = tf.constant([1, 5, 3, 7], shape = [2,2])

print("Matrix B : \n", session.run(matrixB))

# Creating Constant Matrix
matrixC = tf.constant([2, 2, 6, 5], shape = [2,2])

print("Matrix C : \n", session.run(matrixC))

# a ^ 2
matrixA2 = tf.pow(matrixA, 2)

# a ^ 2 + b
matrixAB = tf.add(matrixA2, matrixB)

# (a ^ 2 + b ) * c
matrixABC = tf.multiply(matrixAB, matrixC)

print("(a ^ 2 + b ) * c Result : \n ", session.run(matrixABC))

session.close()
