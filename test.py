import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym.neuralgym as ng

from inpaint_model import InpaintCAModel
from timeit import default_timer as timer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image", default="", type=str, help="The filename of the image to be completed."
)
parser.add_argument(
    "--mask", default="", type=str, help="The filename of the mask, value 255 indicates mask."
)
parser.add_argument(
    "--reference",
    default="",
    type=str,
    help="The filename of the reference, value 255 indicates mask.",
)
parser.add_argument("--output", default="output.png", type=str, help="Where to write output.")
parser.add_argument(
    "--checkpoint_dir", default="", type=str, help="The directory of tensorflow checkpoint."
)


if __name__ == "__main__":
    FLAGS = ng.Config("inpaint.yml")
    # ng.get_gpus(1)
    args, unknown = parser.parse_known_args()

    model = InpaintCAModel()
    image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    mask = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
    reference = cv2.imread(args.reference, cv2.IMREAD_UNCHANGED)
    reference = cv2.cvtColor(reference, cv2.COLOR_BGRA2RGBA)

    assert image.shape == mask.shape and image.shape == reference.shape

    h, w, _ = image.shape
    grid = 8
    image = image[: h // grid * grid, : w // grid * grid, :]
    mask = mask[: h // grid * grid, : w // grid * grid, :]
    reference = reference[: h // grid * grid, : w // grid * grid, :]
    print("Shape of image: {}".format(image.shape))
    print("Shape of reference: {}".format(reference.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    reference = np.expand_dims(reference, 0)
    input_image = np.concatenate([image, mask, reference], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output, summary = model.build_server_graph(FLAGS, input_image)
        output = (output + 1.0) * 127.5
        summary = (output + 1.0) * 127.5

        # Load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print("Model loaded.")
        start = timer()
        result_out, result_summary = sess.run([output, summary])
        end = timer()
        print(end - start)
        output_image = cv2.cvtColor(result_out[0], cv2.COLOR_BGRA2RGBA)
        summary_image = cv2.cvtColor(result_summary[0], cv2.COLOR_BGRA2RGBA)
        cv2.imwrite(args.output, output_image)
        cv2.imwrite("summary_" + args.output, summary_image)
