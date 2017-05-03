import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
from time import time
from sklearn.utils import shuffle

def CNN(x, dropout_prob):
	#hyperparameters
	mu = 0
	sigma = 0.1
	# All weights are initialised with values close to zero from the normal dist
	# Layer 1.
	# Convolutional layer, 1 step kernel stride with relu activation. Input = 32x32x3. Output = 28x28x108. 
	w1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 108), mean = mu, stddev = sigma ,name="w1"))
	b1 = tf.Variable(tf.zeros(108), name = "b1")
	conv1 = tf.nn.relu(tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID', name="conv1") + b1, name="relu1")
	# Pooling layer, 2 x 2 kernel size and stride.  Input = 28x28x108. Output = 14x14x108.
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool1")
	# Local response normalization https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
	lrn1 = tf.nn.lrn(pool1, 3, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="lrn1")

	# Layer 2.
	# Convolutional layer, 1 step kernel stride with relu activation. Input = 14x14x108. Output = 10x10x200.
	w2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 108, 200), mean = mu, stddev = sigma),name="w1")
	b2 = tf.Variable(tf.zeros(200), name = "b2")
	conv2  = tf.nn.relu(tf.nn.conv2d(lrn1, w2, strides=[1, 1, 1, 1], padding='VALID', name="conv2") + b2, name="relu2")
	# Pooling layer, 2 x 2 kernel size and stride.  Input = 10x10x200. Output = 5x5x200.
	pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="pool2")
	# Local response normalization https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
	lrn2 = tf.nn.lrn(pool2, 3, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="lrn2") 

	# Flatten.
	# Flatten each convolution and concat. flatten lrn1,lrn2 = 5292, 5000
	flatten_lrn2   = flatten(lrn2) 
	flatten_lrn1 = flatten(tf.nn.max_pool(lrn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'))
	# concatenated layer size = 10292
	fc0 = tf.concat([flatten_lrn1,flatten_lrn2], 1)

	# Layer 3. 
	# Fully Connected layer with relu activation. Input = 10292. Output = 200.
	fc1_W = tf.Variable(tf.truncated_normal(shape=(10292, 200), mean = mu, stddev = sigma))
	fc1_b = tf.Variable(tf.zeros(200))
	fc1   = tf.nn.relu(tf.matmul(fc0, fc1_W) + fc1_b)

	# dropout 
	dropfc1 = tf.nn.dropout(fc1, dropout_prob)

	# Layer 4: Fully Connected. Input = 200. Output = 100.
	fc2_W  = tf.Variable(tf.truncated_normal(shape=(200, 100), mean = mu, stddev = sigma))
	fc2_b  = tf.Variable(tf.zeros(100))
	fc2    = tf.nn.relu(tf.matmul(dropfc1, fc2_W) + fc2_b)

	# Layer 5: Fully Connected. Input = 100. Output = 43.
	fc3_W  = tf.Variable(tf.truncated_normal(shape=(100, 43), mean = mu, stddev = sigma))
	fc3_b  = tf.Variable(tf.zeros(43))
	logits = tf.matmul(fc2, fc3_W) + fc3_b
	# print "logits",logits.shape

	return logits


def split_dataset(x_data,y_data):
	total_num_data = len(y_data)
	# X_shuffled, y_shuffled = shuffle(x_data, y_data)
	#Data splitting
	np.random.seed(seed=123) #Set seed for reproducability
	#Size of the train subset
	size_train = 0.8 #80% of the given dataset
	size_train = int(size_train*total_num_data)
	#Generate a mask and subsample training and validation da tasets
	mask=np.random.permutation(np.arange(total_num_data))[:size_train]
	# sub are training data
	X_train_sub, y_train_sub = x_data[mask], y_data[mask]
	y_train_sub = np.squeeze(y_train_sub[:,0]).astype(np.int32)
	# validation data
	X_val, y_val = np.delete(X_train, mask,0), np.delete(y_train, mask,0)
	y_val = np.squeeze(y_val[:,:,0]).astype(np.int32)
	print "-"*20; print "splitting dataset into train and evaluation"
	print "Number of images in the train dataset = ", len(y_train_sub)
	print "Number of images in the validation dataset = ", len(y_val)
	print "X_train_sub.shape, y_train_sub.shape, X_val.shape, y_val.shape"
	print X_train_sub.shape, y_train_sub.shape, X_val.shape, y_val.shape;
	print "-"*20;	
	# print "X_train_sub.shape, y_train_sub.shape, X_val.shape, y_val.shape"
	# print X_train_sub.shape, y_train_sub.shape, X_val.shape, y_val.shape;
	# print y_val[120:130]
	# print "-"*20
	return X_train_sub, y_train_sub, X_val, y_val


def evaluate(X_data, y_data):
	X_data, y_data = shuffle(X_data, y_data)
	total_accuracy = 0
	BATCH_SIZE = 128
	num_examples = len(X_data)
	sess = tf.get_default_session()
	for offset in range(0, num_examples, BATCH_SIZE):
		batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
		accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, dropout_prob:1.0})
		total_accuracy += (accuracy * len(batch_x))
		eval_return = total_accuracy / num_examples
	return eval_return


x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43);#print one_hot_y
rate = tf.placeholder(tf.float32, shape=[])
dropout_prob = tf.placeholder(tf.float32,shape=[])
logits = CNN(x, dropout_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
#Model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
###############################################################################################
print "-"*10
trainingData = np.load('trainingData_dict.npz', 'r')
# testData = np.load('testData_dict.npz', 'r')
print "trainingData keys: " + str(trainingData.files)
print "length of data trainingData[labels] " + str(len(trainingData["labels"]))
X_train = trainingData["features"]
# X_train_ini = np.reshape(X_train,(len(trainingData["features"]),32,32,3)).astype(np.float32)#*255
X_train_ini = X_train[:,:,:]
X_train_ini = np.reshape(X_train,(len(trainingData["features"]),32,32,3)).astype(np.float32)#*255
y_train = trainingData["labels"]
y_train_ini = np.squeeze(y_train[:,:,:]).astype(np.int32)
print "length, type and shape of x train features",len(X_train_ini),X_train_ini.dtype ,X_train_ini.shape
print "length, type and shape of y train labels",len(y_train_ini), y_train_ini.dtype ,y_train_ini.shape
print "y sample class label" ,y_train_ini[500]


def main():
	X_train_sub, y_train_sub, X_val, y_val = split_dataset(X_train_ini,y_train_ini)
	EPOCHS = 5
	BATCH_SIZE = 128
	learning_rate = 0.001

	start_time = time()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# saver.restore(sess, tf.train.latest_checkpoint('.'))
		print "-"*10;print "training"
		for i in range(EPOCHS):
			X_train_sub, y_train_sub = shuffle(X_train_sub, y_train_sub)
			# print "length, type and shape of X_train_sub ",len(X_val),X_val.dtype ,X_val.shape
			# print "length, type and shape of y_train_sub",len(y_val), y_val.dtype ,y_val.shape
			# print "y sample class label" ,y_val[500]

			num_examples = len(X_train_sub)
			for offset in range(0, num_examples, BATCH_SIZE):
				end = offset+BATCH_SIZE
				batch_x, batch_y = X_train_sub[offset:end], y_train_sub[offset:end]
				sess.run([training_operation], feed_dict={x:batch_x, y:batch_y,rate:learning_rate,dropout_prob:0.5})#dropout is (keep_prob)
				# print ".", batch_y.shape
			validation_accuracy = evaluate(X_val,y_val)
			print("EPOCH {} ...".format(i+1))
			print("Validation Accuracy = {:.3f}".format(validation_accuracy))
			print()

		# print("Final Validation Accuracy = {:.3f}".format(validation_accuracy))
		saver.save(sess, 'CNN')
		print("Model saved")


	end_time = time()
	time_taken = end_time - start_time # time_taken is in seconds

	hours, rest = divmod(time_taken,3600)
	minutes, seconds = divmod(rest, 60)

	print ("Time: ", hours, "h, ", minutes, "min, ", seconds, "s ")
	sess.close()


# sess.run(training_operation, feed_dict={x: X_train_sub, y: y_train_sub, rate: 0.5, dropout_prob: 0.5})





	# with tf.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	num_examples = len(X_train_ini)

	# 	print("Training...")
	# 	print()
	# 	for i in range(EPOCHS):
	# 		X_tref, y_tref = shuffle(X_train_ini, y_train_ini)

	# 		for offset in range(0, num_examples, BATCH_SIZE):
	# 			end = offset + BATCH_SIZE
	# 			batch_x, batch_y = X_tref[offset:end], y_tref[offset:end]
	# 			sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

	# 		validation_accuracy = evaluate(X_valid_ini, y_valid_ini)
	# 		print("EPOCH {} ...".format(i+1))
	# 		print("Validation Accuracy = {:.3f}".format(validation_accuracy))
	# 		print()

	# 	print("Final Validation Accuracy = {:.3f}".format(validation_accuracy))
	# 	saver.save(sess, 'cnn')
	# 	print("Model saved")




	# with tf.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	num_examples = len(X_train_ini)

	# 	print("Training...")
	# 	print()
	# 	for i in range(EPOCHS):
	# 		X_tref, y_tref = shuffle(X_train_ini, y_train_ini)
	# 		for offset in range(0, num_examples, BATCH_SIZE):
	# 			end = offset + BATCH_SIZE
	# 			batch_x, batch_y = X_tref[offset:end], y_tref[offset:end]
	# 			sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

	# 		validation_accuracy = evaluate(X_valid_ini, y_valid_ini)
	# 		print("EPOCH {} ...".format(i+1))
	# 		print("Validation Accuracy = {:.3f}".format(validation_accuracy))
	# 		print()

	# 	print("Final Validation Accuracy = {:.3f}".format(validation_accuracy))
	# 	saver.save(sess, 'cnn')
	# 	print("Model saved")

	# end_time = time()
	# time_taken = end_time - start_time # time_taken is in seconds
	# hours, rest = divmod(time_taken,3600)
	# minutes, seconds = divmod(rest, 60)
	# print ("Time: ", hours, "h, ", minutes, "min, ", seconds, "s ")


# print "fin"







if __name__ == '__main__':
	main()
	print "ende"