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

if __name__ == "__main__":
	file_name = ""
	model_file = ""

	parser = argparse.ArgumentParser()
	parser.add_argument("--image", help="image to be processed")
	parser.add_argument("--graph", help="graph/model to be executed")
	args = parser.parse_args()

	if args.graph:
		model_file = args.graph
	if args.image:
		file_name = args.image
	
	if args.graph and args.image:

		input_name = "file_reader"
		output_name = "normalized"

		file_reader = tf.read_file(file_name, input_name)

		image_reader = tf.image.decode_png(file_reader, channels = 1, name='png_reader')

		float_caster = tf.cast(image_reader, tf.float32)
		dims_expander = tf.expand_dims(float_caster, 0);
		resized = tf.image.resize_bilinear(dims_expander, [28, 28])
		normalized = tf.divide(tf.subtract(resized, [0]), [255])
		sess = tf.Session()
		t = sess.run(normalized)

		graph = load_graph(model_file)

		input_name = "import/Placeholder"
		output_name = "import/final_result"

		input_operation = graph.get_operation_by_name(input_name);
		output_operation = graph.get_operation_by_name(output_name);

		with tf.Session(graph=graph) as sess:
			start = time.time()
			results = sess.run(output_operation.outputs[0],
				  {input_operation.outputs[0]: t})
			end=time.time()
		results = np.squeeze(results)

		print('Evaluation time: {:.9f}s\n'.format(end-start))

		top_k = results.argsort()[-5:][::-1]

		for i in top_k:
			print(i, results[i])

	else:
		print("ERROR")
		




