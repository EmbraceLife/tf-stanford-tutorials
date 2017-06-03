##############################
### Goal
# create a sequence of data
## tf.linspace(
# 			start = tensor, can be value, vector, matrix, list, array
# 			stop = tensor, must be float32 or float64, last entry
# 			number = tensor, must int32, int64
# 			name = optional
# )

## tf.range(
			# start = 0-D tensor, default 0
			# limit = 0-D tensor, upper limit exclusive
			# delta = 0-D tensor, incremental, default 1,
			# dtype = set resulting tensor type
			# name = default to 'range'
# )

import tensorflow as tf
import numpy as np


g1 = tf.get_default_graph()

with g1.as_default():
	# dt sq1
	sq1 = tf.linspace(start = 10., stop=13., num=8, name='sq1')
	# sq1 as operation on its own can be displayed in tensorboard
	sq2 = tf.range(start=0, limit=5, delta=1, name='sq2')
	sq3 = tf.range(limit=10, name='sq3')

with tf.Session(graph=g1) as sess:
	writer = tf.summary.FileWriter("log/01_make_sequence", sess.graph)
	sess.run(sq1)

writer.close()
