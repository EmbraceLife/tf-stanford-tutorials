import tensorflow as tf

# get official default graph
g1 = tf.get_default_graph()

# create user graph
g2 = tf.Graph()


# using the official default graph to build operations inside
# tensorboard will plot everything defined inside the graph below
with g1.as_default():
	with tf.variable_scope('ops1'):
		a = tf.constant(3)
		c = tf.constant(5)
		op1 = tf.add(a, c)

	with tf.variable_scope('ops3-5'):
		x = 2
		y = 3
		op3 = tf.add(x, y)
		op4 = tf.multiply(x, y)
		useless = tf.multiply(x, op1)
		op5 = tf.pow(op3, op4)


# add ops to the user created graph
with g2.as_default():
	b = tf.constant(5)
	d = tf.constant(6)
	op2 = tf.add(b, d)
	# op3 = tf.add(a, b) # item must from the same graph


# using the official graph
sess = tf.Session(graph=g1)
sess.run(op1)
writer = tf.summary.FileWriter('./log/01_default_op1', sess.graph)
sess.close()


# using the user graph
sess = tf.Session(graph=g2)
writer = tf.summary.FileWriter('./log/01_user', sess.graph)
sess.run(op2)
sess.close()


with tf.Session(graph=g1) as sess:
	# default_op5 is the same as default_op1
	writer = tf.summary.FileWriter('./log/01_default_op5', sess.graph)
	op5, not_useless = sess.run([op5, useless])
