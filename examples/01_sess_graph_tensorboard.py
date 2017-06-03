import tensorflow as tf

# get official default graph
g1 = tf.get_default_graph()
# create user graph
g2 = tf.Graph()

# using the official default graph to build operations inside
# tensorboard will plot everything defined inside the graph below
with g1.as_default():
	# build a scope box to separate ops
	with tf.variable_scope('ops1'):
		a = tf.constant(3, name="a")
		c = tf.constant(5, name="c")
		op1 = tf.add(a, c, name="add")

	with tf.variable_scope('ops3-5'):
		x = 2 # everything inside graph is made a tensor (constant in this case)
		y = 3
		op3 = tf.add(x, y)
		op4 = tf.multiply(x, y, name="add2")
		useless = tf.multiply(x, op1, name="multiply")
		op5 = tf.pow(op3, op4)


# add ops to the user created graph
with g2.as_default():
	b = tf.constant(5, name="b")
	d = tf.constant(6, name="d")
	op2 = tf.add(b, d, name="add3")
	# op3 = tf.add(a, b) # item must from the same graph


# using the official graph
sess = tf.Session(graph=g1)
# Create the summary writer after graph definition and before running your session
writer = tf.summary.FileWriter('./log/01_default_op1', sess.graph)
sess.run(op1) # op3-5 is not run, but still on graph display
sess.close()
# todo: when do we really need to use it?
writer.close()


# using the user graph
sess = tf.Session(graph=g2)
writer = tf.summary.FileWriter('./log/01_user', sess.graph)
sess.run(op2)
sess.close()
writer.close()
