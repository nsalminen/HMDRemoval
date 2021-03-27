import os
import glob
from random import randint

import tensorflow as tf
import keras
import tempfile
import neuralgym.neuralgym as ng
from keras_vggface.vggface import VGGFace
from keras import backend as K

from inpaint_model import InpaintCAModel


def multigpu_graph_def(model, FLAGS, image_ref_mask_data, identity_model, gpu_id=0, loss_type="g"):
    with tf.device("/cpu:0"):
        images, references, masks = image_ref_mask_data.data_pipeline(FLAGS.batch_size)
    if gpu_id == 0 and loss_type == "g":
        _, _, losses = model.build_graph_with_losses(
            FLAGS, images, masks, references, identity_model, FLAGS, summary=True, reuse=True
        )
    else:
        _, _, losses = model.build_graph_with_losses(
            FLAGS, images, masks, references, identity_model, FLAGS, reuse=True
        )
    if loss_type == "g":
        return losses["g_loss"]
    elif loss_type == "d":
        return losses["d_loss"]
    else:
        raise ValueError("loss type is not supported.")


if __name__ == "__main__":
    FLAGS = ng.Config("inpaint.yml")

    slurm_id = os.environ.get("SLURM_JOB_ID", None)
    if slurm_id:
        FLAGS.log_dir += "_" + str(slurm_id)

    img_shapes = FLAGS.img_shapes
    masks = None
    references = None
    identity_model = None
    vggface_weights = None
    tf_checkpoint_path = None

    tf.set_random_seed(int(slurm_id))

    if FLAGS.guided:
        raise NotImplementedError("{} not implemented.".format("guides"))

    # Load and save VGGFace model weights for restoration at training start
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if FLAGS.identity_loss:
        with tf.Session(config=config) as sess:
            with tf.variable_scope("VGGFace"):
                with tf.variable_scope("model"):
                    vggface = VGGFace(
                        model="resnet50", include_top=False, input_shape=(224, 224, 3)
                    )
                    add_16_out7 = vggface.layers[-3].output

                    identity_model = keras.Model(vggface.input, [add_16_out7])
                    for layer in identity_model.layers:
                        layer.trainable = False

            vggface_weights = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="VGGFace/model"
            )
            with tempfile.NamedTemporaryFile() as f:
                tf_checkpoint_path = tf.train.Saver(vggface_weights).save(sess, f.name)

    # Training data
    if FLAGS.custom_mask:
        # Read training image paths
        with open(FLAGS.data_flist[FLAGS.dataset][0]) as f:
            fnames_images = f.read().splitlines()
        # Read training masks paths
        with open(FLAGS.data_flist[FLAGS.dataset][2]) as f:
            fnames_masks = f.read().splitlines()
        # Read training reference paths
        with open(FLAGS.data_flist[FLAGS.dataset][4]) as f:
            fnames_refs = f.read().splitlines()

        fnames_images_refs_masks = list(zip(fnames_images, fnames_refs, fnames_masks))

        image_ref_mask_data = ng.data.DataFromFNames(
            fnames_images_refs_masks,
            [img_shapes, img_shapes, img_shapes],
            random_crop=FLAGS.random_crop,
            nthreads=FLAGS.num_cpus_per_job,
        )

        images, references, masks = image_ref_mask_data.data_pipeline(FLAGS.batch_size)
        print("masks.shape", masks)
    else:
        with open(FLAGS.data_flist[FLAGS.dataset][0]) as f:
            fnames = f.read().splitlines()
        if FLAGS.guided:
            fnames = [(fname, fname[:-4] + "_edge.jpg") for fname in fnames]
            img_shapes = [img_shapes, img_shapes]
        data = ng.data.DataFromFNames(
            fnames, img_shapes, random_crop=FLAGS.random_crop, nthreads=FLAGS.num_cpus_per_job
        )
        images = data.data_pipeline(FLAGS.batch_size)

    # Main model
    model = InpaintCAModel()
    g_vars, d_vars, losses = model.build_graph_with_losses(
        FLAGS, images, masks, references, identity_model
    )

    # Validation images
    if FLAGS.val:
        with open(FLAGS.data_flist[FLAGS.dataset][1]) as f:
            val_fnames = f.read().splitlines()
        with open(FLAGS.data_flist[FLAGS.dataset][3]) as f:
            val_mask_fnames = f.read().splitlines()
        with open(FLAGS.data_flist[FLAGS.dataset][5]) as f:
            val_ref_fnames = f.read().splitlines()
        if FLAGS.guided:
            val_fnames = [(fname, fname[:-4] + "_edge.jpg") for fname in val_fnames]
        # Progress monitor by visualizing static images
        for i in range(FLAGS.static_view_size):
            static_fnames = val_fnames[i : i + 1]
            static_mask_fnames = val_mask_fnames[i : i + 1]
            static_ref_fnames = val_ref_fnames[i : i + 1]
            static_fnames = list(zip(static_fnames, static_mask_fnames, static_ref_fnames))
            static_images, static_masks, static_refs = ng.data.DataFromFNames(
                static_fnames,
                [img_shapes, img_shapes, img_shapes],
                nthreads=8,
                random_crop=FLAGS.random_crop,
            ).data_pipeline(1)
            static_inpainted_images = model.build_static_infer_graph(
                FLAGS, static_images, static_masks, static_refs, name="static_view/%d" % i
            )
    # Training settings
    lr = tf.get_variable(
        "lr", shape=[], trainable=False, initializer=tf.constant_initializer(1e-4)
    )
    d_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999)
    g_optimizer = d_optimizer
    # Train discriminator with secondary trainer, should initialize before
    # primary trainer.
    # discriminator_training_callback = ng.callbacks.SecondaryTrainer(
    discriminator_training_callback = ng.callbacks.SecondaryMultiGPUTrainer(
        num_gpus=FLAGS.num_gpus_per_job,
        pstep=1,
        optimizer=d_optimizer,
        var_list=d_vars,
        max_iters=1,
        grads_summary=False,
        graph_def=multigpu_graph_def,
        graph_def_kwargs={
            "model": model,
            "FLAGS": FLAGS,
            "image_ref_mask_data": image_ref_mask_data,
            "identity_model": identity_model,
            "loss_type": "d",
        },
    )
    # Train generator with primary trainer
    # trainer = ng.train.Trainer(
    trainer = ng.train.MultiGPUTrainer(
        num_gpus=FLAGS.num_gpus_per_job,
        optimizer=g_optimizer,
        var_list=g_vars,
        max_iters=FLAGS.max_iters,
        graph_def=multigpu_graph_def,
        grads_summary=False,
        gradient_processor=None,
        graph_def_kwargs={
            "model": model,
            "FLAGS": FLAGS,
            "image_ref_mask_data": image_ref_mask_data,
            "identity_model": identity_model,
            "loss_type": "g",
        },
        spe=FLAGS.train_spe,
        log_dir=FLAGS.log_dir,
    )
    # Add all callbacks
    trainer.add_callbacks(
        [
            discriminator_training_callback,
            ng.callbacks.IdentityModelRestorer(vggface_weights, tf_checkpoint_path),
            ng.callbacks.WeightsViewer(),
            ng.callbacks.ModelRestorer(
                trainer.context["saver"],
                dump_prefix=FLAGS.model_restore + "/snap",
                optimistic=True,
            ),
            ng.callbacks.ModelSaver(
                FLAGS.train_spe, trainer.context["saver"], FLAGS.log_dir + "/snap"
            ),
            ng.callbacks.SummaryWriter(
                (FLAGS.val_psteps // 1), trainer.context["summary_writer"], tf.summary.merge_all()
            ),
        ]
    )
    # Launch training
    trainer.train()
