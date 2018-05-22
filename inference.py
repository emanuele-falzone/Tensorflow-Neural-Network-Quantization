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

def load_image(file_name):
	file_reader = tf.read_file(file_name, "file_reader")

	image_reader = tf.image.decode_png(file_reader, channels = 1, name='png_reader')

	float_caster = tf.cast(image_reader, tf.float32)
	dims_expander = tf.expand_dims(float_caster, 0);
	resized = tf.image.resize_bilinear(dims_expander, [28, 28])
	normalized = tf.divide(tf.subtract(resized, [0]), [255])

	sess = tf.Session()
	return sess.run(normalized)

def predict(graph, image):
	input_operation = graph.get_operation_by_name("import/Placeholder");
	output_operation = graph.get_operation_by_name("import/final_result");
	
	with tf.Session(graph=graph) as sess:
		for i in range(0,10):
			start = time.time()
			results = sess.run(output_operation.outputs[0],
		  		{input_operation.outputs[0]: image})
			end=time.time()
			print('{:.9f}s  '.format(end-start))
	results = np.squeeze(results)
	return results, end-start

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--image", help="image to be processed")
	parser.add_argument("--graph", help="graph/model to be executed")
	args = parser.parse_args()

	if args.graph and args.image:

		image = load_image(args.image)
		graph = load_graph(args.graph)

		results, prediction_time = predict(graph, image)

		#print('Evaluation time: {:.9f}s\n'.format(prediction_time))

		top_k = results.argsort()[-5:][::-1]

		for i in top_k:
			print(i, results[i])

	else:
		parser.print_help()
