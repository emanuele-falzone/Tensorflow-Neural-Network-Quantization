import tensorflow as tf
import numpy as np
import time
import argparse

def load_graph(model_file):
	graph = tf.Graph()
	graph_def = tf.GraphDef()

	with open(model_file, "rb") as f:
		graph_def.ParseFromString(f.read())
	with graph.as_default():
		tf.import_graph_def(graph_def)

	return graph

def predict(graph, image):
	input_operation = graph.get_operation_by_name("import/Placeholder");
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
	parser.add_argument("--times", help="number of images to be processed")
	parser.add_argument("--graph", help="graph/model to be executed")
	args = parser.parse_args()

	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	eval_data = mnist.test.images # Returns np.array
	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

	images = len(eval_labels)

	if args.times:
		images = int(args.times)

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

