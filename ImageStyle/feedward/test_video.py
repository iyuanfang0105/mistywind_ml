import tensorflow as tf
import cv2
import numpy as np
import utils



def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def infer(sess, x_tensor, y_tensor, image):
    return sess.run(y_tensor, feed_dict={x_tensor: image})


if __name__ == '__main__':
    import time
    image_max_size = 400
    content_img_path = '../../test_data/me.jpeg'

    content_img = utils.read_image(content_img_path, preprocess_flag=True, max_size=image_max_size)

    graph = load_graph('test/aa.pb')

    x = graph.get_tensor_by_name('prefix/img_placeholder:0')
    y = graph.get_tensor_by_name('prefix/Tanh:0')

    st = time.time()

    with tf.Session(graph=graph) as sess:
        cap = cv2.VideoCapture(0)
        index = 0
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_resize = cv2.resize(frame, (image_max_size, image_max_size)).astype(np.float32)
            frame_resize = utils.preprocess(frame_resize)
            frame_resize = infer(sess, x, y, frame_resize)
            frame_resize = utils.postprocess(frame_resize)
            # Display the resulting frame
            cv2.imshow('frame', frame)
            cv2.imwrite(str(index)+'.jpg', frame_resize)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

        # y_v = infer(sess, x, y, content_img)
    # dt = time.time() - st

    # print(dt)
