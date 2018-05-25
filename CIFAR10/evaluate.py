import os
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf

def unpickle(file):
	import cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict

def load_data():

	path = os.path.join('.', 'cifar-10-batches-py/test_batch')
	batch = unpickle(path)
	images = batch['data'].reshape(10000,3,32,32)
	eval_data = np.transpose(images, (0, 2, 3, 1))
	eval_labels = np.asarray(batch['labels']).astype(dtype=np.int32)
	eval_data= np.true_divide(eval_data,256).astype(dtype=np.float32)

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
		    image = eval_data[i].reshape(1,32,32,3)
		    results, prediction_time = predict(graph, image)
		    predicted = results.argsort()[::-1][0]
		    accuracies.append(float(predicted==eval_labels[i]))
		    times.append(prediction_time)
	    	
		print('               Accuracy: {:.9f}'.format(np.mean(accuracies)))
		print('Average prediction time: {:.9f}s'.format(np.mean(times)))
	else:
		parser.print_help()

