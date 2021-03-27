import time
import os, sys
import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym.neuralgym as ng
from tqdm import tqdm

from inpaint_model import InpaintCAModel

parser = argparse.ArgumentParser()
parser.add_argument(
    "--flist",
    default="",
    type=str,
    help="The filenames of image to be processed: input, mask, output.",
)
parser.add_argument(
    "--image_height",
    default=256,
    type=int,
    help="The height of images should be defined, otherwise batch mode is not supported.",
)
parser.add_argument(
    "--image_width",
    default=256,
    type=int,
    help="The width of images should be defined, otherwise batch mode is not" " supported.",
)
parser.add_argument(
    "--summary",
    default=0,
    type=int,
    help="Whether an additional image summary should be written to the output directory.",
)
parser.add_argument(
    "--output_dir",
    default="./output",
    type=str,
    help="The directory that output images should be written to.",
)
parser.add_argument(
    "--checkpoint_dir", default="", type=str, help="The directory of tensorflow checkpoint."
)

if __name__ == "__main__":
    FLAGS = ng.Config("inpaint.yml")
    args = parser.parse_args()
    print("Summary " + str(args.summary))

    result_path = args.output_dir + "/result/"
    summary_path = args.output_dir + "/summary/"

    if os.listdir(args.output_dir):
        sys.exit("\033[41m" + "Specified directory is not empty." + "\033[0m")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if args.summary == 1 and not os.path.exists(summary_path):
        os.makedirs(summary_path)

    ng.get_gpus(1)
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, args.image_height, args.image_width * 3, 4)
    )
    output, summary = model.build_server_graph(FLAGS, input_image_ph)
    output = (output + 1.0) * 127.5
    summary = (summary + 1.0) * 127.5

    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print("Model loaded.")

    with open(args.flist, "r") as f:
        lines = f.read().splitlines()
    t = time.time()
    for line in tqdm(lines):
        image_path, mask_path, ref_path = line.split()
        base = os.path.basename(image_path)

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(
            image, (args.image_height, args.image_width), interpolation=cv2.INTER_NEAREST
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(
            mask, (args.image_height, args.image_width), interpolation=cv2.INTER_NEAREST
        )
        mask = mask[:, :, -1]
        mask = np.repeat(mask[:, :, np.newaxis], 4, axis=2)
        reference = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)
        reference = cv2.resize(
            reference, (args.image_height, args.image_width), interpolation=cv2.INTER_NEAREST
        )
        reference = cv2.cvtColor(reference, cv2.COLOR_BGRA2RGBA)

        assert image.shape == mask.shape and image.shape == reference.shape

        h, w, _ = image.shape
        grid = 8
        image = image[: h // grid * grid, : w // grid * grid, :]
        mask = mask[: h // grid * grid, : w // grid * grid, :]
        reference = reference[: h // grid * grid, : w // grid * grid, :]

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        reference = np.expand_dims(reference, 0)
        input_image = np.concatenate([image, mask, reference], axis=2)

        result_output, result_summary = sess.run(
            [output, summary], feed_dict={input_image_ph: input_image}
        )
        output_path = os.path.join(result_path, base)
        output_image = cv2.cvtColor(result_output[0], cv2.COLOR_BGRA2RGBA)
        cv2.imwrite(output_path, output_image)

        if args.summary == 1:
            summary_output_path = os.path.join(summary_path, base)
            summary_image = cv2.cvtColor(result_summary[0], cv2.COLOR_BGRA2RGBA)
            cv2.imwrite(summary_output_path, summary_image)
