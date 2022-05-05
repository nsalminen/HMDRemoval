import argparse
import glob
import os
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from scipy.spatial import distance
from scipy.ndimage.filters import uniform_filter, gaussian_filter
from scipy import signal
from enum import Enum
from skimage import io
from tqdm import tqdm
from PIL import Image
from skimage.util.dtype import dtype_range
from skimage._shared.utils import warn, check_shape_equality
from skimage.metrics import structural_similarity, mean_squared_error
from inpaint_ops import surface_conv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pred_folder",
    default="./pred/",
    type=str,
    help="The path of the folder containing the predicted images.",
)
parser.add_argument("--channels", default="4", type=int, help="The amount of channels of the images to use.")
parser.add_argument(
    "--flist",
    default="",
    type=str,
    help="The space-separated file for the ground truth and mask image files.",
)
parser.add_argument("--image_size", default="224", type=int, help="The image size for both width and height.")


def extract_face(image, required_size=(160, 160)):
    image_rgb = image[:, :, :3]
    results = detector.detect_faces(image_rgb)
    nrof_faces = len(results)
    if nrof_faces > 0:
        x1, y1, width, height = results[0]["box"]
    else:
        x1 = 24
        y1 = 24
        height = image_rgb.shape[0] - 24 * 2
        width = image_rgb.shape[1] - 24 * 2
        print("Couldn't extract face for " + filename)
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image_rgb[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


def get_facenet_embedding(image):
    pixels = extract_face(image)
    pixels = pixels.astype("float32")
    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / std
    samples = np.expand_dims(pixels, axis=0)
    yhat = facenet_model.predict(samples)
    return yhat[0]


def identity_distance(true, pred):
    true_embedding = get_facenet_embedding(true)
    pred_embedding = get_facenet_embedding(pred)
    return distance.euclidean(true_embedding, pred_embedding)


def vifp(GT, P, sigma_nsq=2):
    """calculates Pixel Based Visual Information Fidelity (vif-p).
    :param GT: first (original) input image.
    :param P: second (deformed) input image.
    :param sigma_nsq: variance of the visual noise (default = 2)
    :returns:  float -- vif-p value.

    Notes
    -----
    .. Author: https://github.com/andrewekhalel/sewar
    """

    class Filter(Enum):
        UNIFORM = 0
        GAUSSIAN = 1

    def filter2(img, fltr, mode="same"):
        return signal.convolve2d(img, np.rot90(fltr, 2), mode=mode)

    def _get_sums(GT, P, win, mode="same"):
        mu1, mu2 = (filter2(GT, win, mode), filter2(P, win, mode))
        return mu1 * mu1, mu2 * mu2, mu1 * mu2

    def _get_sigmas(GT, P, win, mode="same", **kwargs):
        if "sums" in kwargs:
            GT_sum_sq, P_sum_sq, GT_P_sum_mul = kwargs["sums"]
        else:
            GT_sum_sq, P_sum_sq, GT_P_sum_mul = _get_sums(GT, P, win, mode)

        return (
            filter2(GT * GT, win, mode) - GT_sum_sq,
            filter2(P * P, win, mode) - P_sum_sq,
            filter2(GT * P, win, mode) - GT_P_sum_mul,
        )

    def _initial_check(GT, P):
        assert GT.shape == P.shape, "Supplied images have different sizes " + str(GT.shape) + " and " + str(P.shape)
        if GT.dtype != P.dtype:
            msg = "Supplied images have different dtypes " + str(GT.dtype) + " and " + str(P.dtype)
            warnings.warn(msg)

        if len(GT.shape) == 2:
            GT = GT[:, :, np.newaxis]
            P = P[:, :, np.newaxis]

        return GT.astype(np.float64), P.astype(np.float64)

    def fspecial(fltr, ws, **kwargs):
        if fltr == Filter.UNIFORM:
            return np.ones((ws, ws)) / ws ** 2
        elif fltr == Filter.GAUSSIAN:
            x, y = np.mgrid[-ws // 2 + 1 : ws // 2 + 1, -ws // 2 + 1 : ws // 2 + 1]
            g = np.exp(-((x ** 2 + y ** 2) / (2.0 * kwargs["sigma"] ** 2)))
            g[g < np.finfo(g.dtype).eps * g.max()] = 0
            assert g.shape == (ws, ws)
            den = g.sum()
            if den != 0:
                g /= den
            return g
        return None

    def _vifp_single(GT, P, sigma_nsq):
        EPS = 1e-10
        num = 0.0
        den = 0.0
        for scale in range(1, 5):
            N = 2.0 ** (4 - scale + 1) + 1
            win = fspecial(Filter.GAUSSIAN, ws=N, sigma=N / 5)

            if scale > 1:
                GT = filter2(GT, win, "valid")[::2, ::2]
                P = filter2(P, win, "valid")[::2, ::2]

            GT_sum_sq, P_sum_sq, GT_P_sum_mul = _get_sums(GT, P, win, mode="valid")
            sigmaGT_sq, sigmaP_sq, sigmaGT_P = _get_sigmas(
                GT, P, win, mode="valid", sums=(GT_sum_sq, P_sum_sq, GT_P_sum_mul)
            )

            sigmaGT_sq[sigmaGT_sq < 0] = 0
            sigmaP_sq[sigmaP_sq < 0] = 0

            g = sigmaGT_P / (sigmaGT_sq + EPS)
            sv_sq = sigmaP_sq - g * sigmaGT_P

            g[sigmaGT_sq < EPS] = 0
            sv_sq[sigmaGT_sq < EPS] = sigmaP_sq[sigmaGT_sq < EPS]
            sigmaGT_sq[sigmaGT_sq < EPS] = 0

            g[sigmaP_sq < EPS] = 0
            sv_sq[sigmaP_sq < EPS] = 0

            sv_sq[g < 0] = sigmaP_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= EPS] = EPS

            num += np.sum(np.log10(1.0 + (g ** 2.0) * sigmaGT_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1.0 + sigmaGT_sq / sigma_nsq))

        return num / den

    GT, P = _initial_check(GT, P)
    return np.mean([_vifp_single(GT[:, :, i], P[:, :, i], sigma_nsq) for i in range(GT.shape[2])])


def _as_float(image0):
    """
    Promote im1, im2 to nearest appropriate floating point precision.

    Notes
    -----
    .. Taken from skimage.metrics
    """
    float_type = np.result_type(image0.dtype, np.float32)
    image0 = np.asarray(image0, dtype=float_type)
    return image0


def peak_signal_noise_ratio(image_true, image_test, ones_mask, *, data_range=255):
    """
    Compute the peak signal to noise ratio (PSNR) for an image.

    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    data_range : int, optional
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.

    Returns
    -------
    psnr : float
        The PSNR metric.

    Notes
    -----
    .. Adapted from skimage.metrics.peak_signal_noise_ratio by Nels Numan
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_psnr`` to
        ``skimage.metrics.peak_signal_noise_ratio``.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    check_shape_equality(image_true, image_test)

    if len(image_true.shape) == 3:
        channels = image_true.shape[2]
    else:
        channels = 1

    image_true, image_test = _as_float(image_true), _as_float(image_test)

    err = np.sum(np.square(image_true - image_test)) / (ones_mask * channels)
    return 10 * np.log10((data_range ** 2) / err)


def l1_loss(true, pred, ones_mask):
    check_shape_equality(true, pred)
    if len(true.shape) == 3:
        channels = true.shape[2]
    else:
        channels = 1
    return np.sum(np.abs(_as_float(true) - _as_float(pred))) / (ones_mask * channels)


def l2_loss(true, pred, ones_mask):
    check_shape_equality(true, pred)
    if len(true.shape) == 3:
        channels = true.shape[2]
    else:
        channels = 1
    return np.sqrt(np.sum(np.square(_as_float(true) - _as_float(pred))) / (ones_mask * channels))


if __name__ == "__main__":
    args = parser.parse_args()

    sess = tf.Session()
    input_image_ph = tf.placeholder(tf.float32, shape=(1, args.image_size, args.image_size, 1))
    output = surface_conv(input_image_ph)

    detector = MTCNN()
    facenet_model = load_model("facenet_keras.h5")

    pred_folder = os.path.normpath(args.pred_folder)
    filelist_pred = glob.glob(pred_folder + "/*.png")

    filenames = []
    ssim_results_full = []
    psnr_results_full = []
    l1_results_full = []
    l2_results_full = []
    vif_results_full = []

    ssim_results_rgb = []
    psnr_results_rgb = []
    l1_results_rgb = []
    l2_results_rgb = []
    id_results_rgb = []
    vif_results_rgb = []

    ssim_results_depth = []
    psnr_results_depth = []
    l1_results_depth = []
    l2_results_depth = []
    vif_results_depth = []

    ssim_results_sn = []
    psnr_results_sn = []
    l1_results_sn = []
    l2_results_sn = []
    vif_results_sn = []

    with open(args.flist, "r") as f:
        lines = f.read().splitlines()
    assert len(lines) == len(filelist_pred)

    for line in tqdm(lines):
        image_path, mask_path, _ = line.split()
        filename = os.path.basename(image_path)
        filenames.append(filename)

        pred = cv2.imread(os.path.join(pred_folder, filename), cv2.IMREAD_UNCHANGED)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGRA2RGBA)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGBA)
        mask = cv2.resize(mask, (pred.shape[0], pred.shape[1]), interpolation=cv2.INTER_NEAREST)
        mask_count = np.count_nonzero(mask) // mask.shape[2]

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        image = cv2.resize(image, (pred.shape[0], pred.shape[1]), interpolation=cv2.INTER_NEAREST)

        ssim_results_full.append(structural_similarity(image, pred, data_range=255, multichannel=True))
        psnr_results_full.append(peak_signal_noise_ratio(image, pred, mask_count, data_range=255))
        l1_results_full.append(l1_loss(image, pred, mask_count))
        l2_results_full.append(l2_loss(image, pred, mask_count))
        vif_results_full.append(vifp(image, pred))

        ssim_results_rgb.append(structural_similarity(image[:, :, :2], pred[:, :, :2], data_range=255, multichannel=True))
        psnr_results_rgb.append(peak_signal_noise_ratio(image[:, :, :2], pred[:, :, :2], mask_count, data_range=255))
        l1_results_rgb.append(l1_loss(image[:, :, :3], pred[:, :, :3], mask_count))
        l2_results_rgb.append(l2_loss(image[:, :, :3], pred[:, :, :3], mask_count))
        vif_results_rgb.append(vifp(image[:, :, :3], pred[:, :, :3]))
        id_results_rgb.append(identity_distance(image, pred))

        ssim_results_depth.append(structural_similarity(image[:, :, 3], pred[:, :, 3], data_range=255))
        psnr_results_depth.append(peak_signal_noise_ratio(image[:, :, 3], pred[:, :, 3], mask_count, data_range=255))
        l1_results_depth.append(l1_loss(image[:, :, 3], pred[:, :, 3], mask_count))
        l2_results_depth.append(l2_loss(image[:, :, 3], pred[:, :, 3], mask_count))
        vif_results_depth.append(vifp(image[:, :, 3], pred[:, :, 3]))

        sn_image = sess.run([output], feed_dict={input_image_ph: np.expand_dims(image[:, :, 3], axis=[0, -1])})
        sn_pred = sess.run([output], feed_dict={input_image_ph: np.expand_dims(pred[:, :, 3], axis=[0, -1])})
        sn_image = np.squeeze(sn_image) * 255
        sn_pred = np.squeeze(sn_pred) * 255

        ssim_results_sn.append(structural_similarity(sn_image, sn_pred, data_range=255, multichannel=True))
        psnr_results_sn.append(peak_signal_noise_ratio(sn_image, sn_pred, mask_count, data_range=255))
        l1_results_sn.append(l1_loss(sn_image, sn_pred, mask_count))
        l2_results_sn.append(l2_loss(sn_image, sn_pred, mask_count))
        vif_results_sn.append(vifp(sn_image, sn_pred))

    results = pd.DataFrame(
        {
            "filename": filenames,
            "ssim (full)": ssim_results_full,
            "psnr (full)": psnr_results_full,
            "l1 (full)": l1_results_full,
            "l2 (full)": l2_results_full,
            "vif (full)": vif_results_full,
            "ssim (rgb)": ssim_results_rgb,
            "psnr (rgb)": psnr_results_rgb,
            "l1 (rgb)": l1_results_rgb,
            "l2 (rgb)": l2_results_rgb,
            "id (rgb)": id_results_rgb,
            "vif (rgb)": vif_results_rgb,
            "ssim (depth)": ssim_results_depth,
            "psnr (depth)": psnr_results_depth,
            "l1 (depth)": l1_results_depth,
            "l2 (depth)": l2_results_depth,
            "vif (depth)": vif_results_depth,
            "ssim (sn)": ssim_results_sn,
            "psnr (sn)": psnr_results_sn,
            "l1 (sn)": l1_results_sn,
            "l2 (sn)": l2_results_sn,
            "vif (sn)": vif_results_sn,
        }
    )

    results.set_index("filename", inplace=True)
    print("Mean metric values\n-------------------")
    meanVals = results.mean(axis=0)
    print(meanVals)
    print("\nBottom 10 images (based on full)\n---------------------------------")
    print(results.nsmallest(10, ["l2 (full)", "ssim (full)", "psnr (full)", "vif (full)", "l1 (sn)", "id (rgb)"]))

    results.to_csv(args.pred_folder + "../eval_{}.csv".format(os.path.basename(pred_folder)))
    meanVals.to_csv(args.pred_folder + "../mean_{}.csv".format(os.path.basename(pred_folder)))
