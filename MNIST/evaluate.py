import os
import time
import struct
import argparse
import numpy as np
import tensorflow as tf

def unpickle(file):
	import cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict

def load_data():

	#load MNIST data from files

	fname_img_eval = os.path.join(os.getcwd(), 'MNIST-data/t10k-images.idx3-ubyte')
	fname_lbl_eval = os.path.join(os.getcwd(), 'MNIST-data/t10k-labels.idx1-ubyte')

	with open(fname_lbl_eval, 'rb') as flbl:
		magic, num = struct.unpack(">II", flbl.read(8))
		eval_labels = np.fromfile(flbl, dtype=np.int8).astype(dtype=np.int32)

	with open(fname_img_eval, 'rb') as fimg:
		magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
		eval_data = np.fromfile(fimg, dtype=np.uint8).reshape(len(eval_labels), rows, cols, 1).astype(dtype=np.float32)
		eval_data = np.true_divide(eval_data, 256)
	
	return eval_data, eval_labels


def load_graph(model_file):
	graph = tf.Graph()
	graph_def = tf.GraphDef()

	with open(model_file, "rb") as f:
		graph_def.ParseFromString(f.read())
	with graph.as_default():
		tf.import_graph_def(graph_def)

	return graph

def predict(graph, image):
	input_operation = graph.get_operation_by_name("import/input");
	output_operation = graph.get_operation_by_name("import/final_result");
	
	with tf.Session(graph=graph) as sess:
		start = time.time()
		results = sess.run(output_operation.outputs[0],
		  	{input_operation.outputs[0]: image})
		end=time.time()
	results = np.squeeze(results)
	return results, end-start

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--graph", help="graph/model to be executed")
	args = parser.parse_args()

	eval_data, eval_labels = load_data()

	images = len(eval_labels)

	if args.graph:
		print('Evaluating model with %d images' % (images))

		graph = load_graph(args.graph)

		accuracies = []
		times = []

		for i in range(0, images):
		    image = eval_data[i].reshape(1,28,28,1)
		    results, prediction_time = predict(graph, image)
		    predicted = results.argsort()[::-1][0]
		    accuracies.append(float(predicted==eval_labels[i]))
		    times.append(prediction_time)
	    	
		print('               Accuracy: {:.9f}'.format(np.mean(accuracies)))
		print('Average prediction time: {:.9f}s'.format(np.mean(times)))
	else:
		parser.print_help()

