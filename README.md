# Generative RGB-D Face Completion for Head-Mounted Display Removal

An open source framework for generative HMD removal in RGB-D images as described in the paper ["Generative RGB-D Face Completion for Head-Mounted Display Removal"](https://www.researchgate.net/publication/349368479_Generative_RGB-D_Face_Completion_for_Head-Mounted_Display_Removal?utm_source=twitter&rgutm_meta1=eHNsLTk1encwWGFEQmlPdTNkWDZTYWJnNUp4TXdMZm1GY1VSWGcwc1htbjg4MnBKV0VzSzQwNzlWcDFockN2WGgxY2I1Z080ZjQ4cTZUYUtpMFkyWThQWUI5Yz0%3D). This framework builds on the RGB image inpainting framework proposed in ["Free-Form Image Inpainting with Gated Convolution"](https://arxiv.org/abs/1806.03589) by Yu et al.

**The source code of this framework is based on the [source code](https://github.com/JiahuiYu/generative_inpainting) by Yu et al.**

![Qualitative result summary](examples/ResultSummary.png?raw=true "Qualitative results summary, shown for color (RGB), depth (D), and estimated surface normals (SN). For visualization, D is normalized to [0, 1] and displayed with the inferno colormap from the matplotlib package. The normal vectors (x, y, z) for each pixel in SN are estimated based on D and are visualized with RGB values.")
_Qualitative results summary. Shown for color (RGB), depth (D), and estimated surface normals (SN). For visualization, D is normalized to [0, 1] and displayed with the inferno colormap from the matplotlib package. The normal vectors (x, y, z) for each pixel in SN are estimated based on D and are visualized with RGB values._

## Run

0. Requirements:
   - Install Python (v3.6).
   - Install the requirements listed in `requirements.txt`.
      - Run `pip install -r requirements.txt` to install with pip.
1. Training:
   - Prepare training images file list with [create_flist.py](/create_flist.py) (needs modification based on file name structure).
   - Modify [inpaint.yml](/inpaint.yml) to set `DATA_FLIST`, `LOG_DIR`, `IMG_SHAPES` and other parameters.
   - Run `python train.py`.
2. Resume training:
   - Modify `MODEL_RESTORE` flag in [inpaint.yml](/inpaint.yml). E.g., `MODEL_RESTORE: hmdRemoval_341953`.
   - Run `python train.py`.
3. Testing:
   - Run `python test.py --image data/input.png --mask data/mask.png --reference data/reference.png --output examples/output.png --checkpoint logs/hmdRemoval_341953`.
4. Still have questions?
   - If you still have questions (e.g.: How to use multi-gpus? How to do batch testing?), please first search over [closed issues in the repo of the base framework](https://github.com/JiahuiYu/generative_inpainting/issues?q=is%3Aissue+is%3Aclosed). If the problem is not solved, please feel free to open an issue.

## TensorBoard

Visualization on [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) for training and validation is supported. Run `tensorboard --logdir model_logs --port 6006` to view training progress.

## License

CC 4.0 Attribution-NonCommercial International

The software is for educational and academic research purpose only.

## Citing

```bibtex
@inproceedings{numan2021generative,
  title={Generative RGB-D Face Completion for Head-Mounted Display Removal},
  author={Numan, Nels and ter Haar, Frank and Cesar, Pablo},
  booktitle={2021 IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (VRW)},
  pages={109--116},
  year={2021},
  organization={IEEE}
}
````
