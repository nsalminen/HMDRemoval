import argparse
import os
import sys
from random import shuffle
import re

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", default="./data", type=str, help="The input folder path")
parser.add_argument(
    "--period_replacement",
    default="data/",
    type=str,
    help="Replacement for first period in input_path",
)
parser.add_argument(
    "--output_path", default="./data_flist", type=str, help="The output folder path"
)
parser.add_argument("--is_sorted", default="1", type=int, help="Whether to sort the output.")


def split_by_numbers(string_):
    return [
        int(s) if s.isdigit() else s for s in re.split(r"(\d+)", os.path.split(string_)[-1]) if s
    ]


def get_mask_list(path_list):
    output_list = []
    for path in path_list:
        filename = os.path.basename(path)
        pose = re.search("pose(.+?)ambient", filename).group(1)
        mask_item_path = input_path + "/mask/mask-pose" + pose + "rgba_850_60.png"
        if not os.path.isfile(mask_item_path):
            sys.exit("\033[41m" + "Mask for " + item_path + " does not exist!" + "\033[0m")
        output_list.append(mask_item_path)
    return output_list


if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    else:
        sys.exit("\033[41m" + "Output folder already exists." + "\033[0m")

    if args.input_path[0] is not ".":
        sys.exit("\033[41m" + "Please run this script directly from a directory." + "\033[0m")

    input_path = args.input_path.replace(".", args.period_replacement, 1)

    train_face_file_paths = []
    train_ref_file_paths = []

    val_face_file_paths = []
    val_ref_file_paths = []

    test_face_file_paths = []
    test_ref_file_paths = []

    train_face_folder = os.listdir(input_path + "/train/input/")
    for item_name in train_face_folder:
        if item_name[0] is ".":
            continue
        item_path = input_path + "/train/input/" + item_name
        train_face_file_paths.append(item_path)

    train_ref_folder = os.listdir(input_path + "/train/ref/")
    for item_name in train_ref_folder:
        if item_name[0] is ".":
            continue
        item_path = input_path + "/train/ref/" + item_name
        train_ref_file_paths.append(item_path)

    val_face_folder = os.listdir(input_path + "/val/input/")
    for item_name in val_face_folder:
        if item_name[0] is ".":
            continue
        item_path = input_path + "/val/input/" + item_name
        val_face_file_paths.append(item_path)

    val_ref_folder = os.listdir(input_path + "/val/ref/")
    for item_name in val_ref_folder:
        if item_name[0] is ".":
            continue
        item_path = input_path + "/val/ref/" + item_name
        val_ref_file_paths.append(item_path)

    test_face_folder = os.listdir(input_path + "/test/input/")
    for item_name in test_face_folder:
        if item_name[0] is ".":
            continue
        item_path = input_path + "/test/input/" + item_name
        test_face_file_paths.append(item_path)

    test_ref_folder = os.listdir(input_path + "/test/ref/")
    for item_name in test_ref_folder:
        if item_name[0] is ".":
            continue
        item_path = input_path + "/test/ref/" + item_name
        test_ref_file_paths.append(item_path)

    if args.is_sorted == 1:
        train_face_file_paths.sort(key=split_by_numbers)
        val_face_file_paths.sort(key=split_by_numbers)
        test_face_file_paths.sort(key=split_by_numbers)

        train_ref_file_paths.sort(key=split_by_numbers)
        val_ref_file_paths.sort(key=split_by_numbers)
        test_ref_file_paths.sort(key=split_by_numbers)

    train_mask_file_paths = get_mask_list(train_face_file_paths)
    val_mask_file_paths = get_mask_list(val_face_file_paths)
    test_mask_file_paths = get_mask_list(test_face_file_paths)

    f = open(os.path.join(args.output_path + "/train_face.flist"), "w")
    f.write("\n".join(train_face_file_paths))
    f.close()
    f = open(os.path.join(args.output_path + "/train_ref.flist"), "w")
    f.write("\n".join(train_ref_file_paths))
    f.close()
    f = open(os.path.join(args.output_path + "/train_mask.flist"), "w")
    f.write("\n".join(train_mask_file_paths))
    f.close()

    f = open(os.path.join(args.output_path + "/val_face.flist"), "w")
    f.write("\n".join(val_face_file_paths))
    f.close()
    f = open(os.path.join(args.output_path + "/val_ref.flist"), "w")
    f.write("\n".join(val_ref_file_paths))
    f.close()
    f = open(os.path.join(args.output_path + "/val_mask.flist"), "w")
    f.write("\n".join(val_mask_file_paths))
    f.close()

    test_face_mask_file_paths = [
        " ".join(map(str, i)) for i in zip(test_face_file_paths, test_mask_file_paths)
    ]
    f = open(os.path.join(args.output_path + "/test_face_mask.flist"), "w")
    f.write("\n".join(test_face_mask_file_paths))
    f.close()
    test_face_mask_ref_file_paths = [
        " ".join(map(str, i)) for i in zip(test_face_mask_file_paths, test_ref_file_paths)
    ]
    f = open(os.path.join(args.output_path + "/test_face_mask_ref.flist"), "w")
    f.write("\n".join(test_face_mask_ref_file_paths))
    f.close()

    print("Written file is: ", args.output_path)
