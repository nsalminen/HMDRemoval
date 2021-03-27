""" common model for DCGAN """
import logging

import cv2
import numpy as np
import neuralgym as ng
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from neuralgym.neuralgym.models import Model
from neuralgym.neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.neuralgym.ops.summary_ops import gradients_summary
from neuralgym.neuralgym.ops.layers import flatten, resize
from neuralgym.neuralgym.ops.gan_ops import gan_hinge_loss, gan_identity_loss

from inpaint_ops import gen_conv, gen_deconv, dis_conv, surface_conv
from inpaint_ops import random_bbox, bbox2mask, local_patch, brush_stroke_mask
from inpaint_ops import resize_mask_like, contextual_attention


logger = logging.getLogger()


class InpaintCAModel(Model):
    def __init__(self):
        super().__init__("InpaintCAModel")

    def build_inpaint_net(
        self,
        x,
        mask,
        ref,
        reuse=False,
        training=True,
        padding="SAME",
        surface_attention=True,
        name="inpaint_net",
    ):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
            ref: reference image [-1, 1]
        Returns:
            [-1, 1] as predicted image
        """
        xin = x
        offset_flow = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        if ref is not None:
            x = tf.concat([x, ones_x, ones_x * mask, ref], axis=3)
        else:
            x = tf.concat([x, ones_x, ones_x * mask], axis=3)

        # Two-stage coarse-to-fine generator network
        cnum = 48
        with tf.variable_scope(name, reuse=reuse), arg_scope(
            [gen_conv, gen_deconv], training=training, padding=padding
        ):
            # Stage 1
            x = gen_conv(x, cnum, 5, 1, name="conv1")
            x = gen_conv(x, 2 * cnum, 3, 2, name="conv2_downsample")
            x = gen_conv(x, 2 * cnum, 3, 1, name="conv3")
            x = gen_conv(x, 4 * cnum, 3, 2, name="conv4_downsample")
            x = gen_conv(x, 4 * cnum, 3, 1, name="conv5")
            x = gen_conv(x, 4 * cnum, 3, 1, name="conv6")
            mask_s = resize_mask_like(mask, x)
            x = gen_conv(x, 4 * cnum, 3, rate=2, name="conv7_atrous")
            x = gen_conv(x, 4 * cnum, 3, rate=4, name="conv8_atrous")
            x = gen_conv(x, 4 * cnum, 3, rate=8, name="conv9_atrous")
            x = gen_conv(x, 4 * cnum, 3, rate=16, name="conv10_atrous")
            x = gen_conv(x, 4 * cnum, 3, 1, name="conv11")
            x = gen_conv(x, 4 * cnum, 3, 1, name="conv12")
            x = gen_deconv(x, 2 * cnum, name="conv13_upsample")
            x = gen_conv(x, 2 * cnum, 3, 1, name="conv14")
            x = gen_deconv(x, cnum, name="conv15_upsample")
            x = gen_conv(x, cnum // 2, 3, 1, name="conv16")
            x = gen_conv(x, 4, 3, 1, activation=None, name="conv17")
            x = tf.nn.tanh(x)
            x_stage1 = x

            # Stage 2, paste result as input
            x = x * mask + xin[:, :, :, 0:4] * (1.0 - mask)
            x.set_shape(xin[:, :, :, 0:4].get_shape().as_list())
            # Conv branch
            # xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)
            xnow = x
            if ref is not None:
                xnow_ref = tf.concat([x, ref], axis=3)
            else:
                xnow_ref = x
            x = gen_conv(xnow_ref, cnum, 5, 1, name="xconv1")
            x = gen_conv(x, cnum, 3, 2, name="xconv2_downsample")
            x = gen_conv(x, 2 * cnum, 3, 1, name="xconv3")
            x = gen_conv(x, 2 * cnum, 3, 2, name="xconv4_downsample")
            x = gen_conv(x, 4 * cnum, 3, 1, name="xconv5")
            x = gen_conv(x, 4 * cnum, 3, 1, name="xconv6")
            x = gen_conv(x, 4 * cnum, 3, rate=2, name="xconv7_atrous")
            x = gen_conv(x, 4 * cnum, 3, rate=4, name="xconv8_atrous")
            x = gen_conv(x, 4 * cnum, 3, rate=8, name="xconv9_atrous")
            x = gen_conv(x, 4 * cnum, 3, rate=16, name="xconv10_atrous")
            x_hallu = x

            # Attention branch
            if surface_attention:
                x_depth = tf.expand_dims(xnow[:, :, :, 3], axis=-1)
                x_sn = surface_conv((x_depth + 1.0) * 127.5)
                xnow = tf.concat([xnow, x_sn], axis=3)
            x = gen_conv(xnow, cnum, 5, 1, name="pmconv1")
            x = gen_conv(x, cnum, 3, 2, name="pmconv2_downsample")
            x = gen_conv(x, 2 * cnum, 3, 1, name="pmconv3")
            x = gen_conv(x, 4 * cnum, 3, 2, name="pmconv4_downsample")
            x = gen_conv(x, 4 * cnum, 3, 1, name="pmconv5")
            x = gen_conv(x, 4 * cnum, 3, 1, name="pmconv6", activation=tf.nn.relu)
            x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
            x = gen_conv(x, 4 * cnum, 3, 1, name="pmconv9")
            x = gen_conv(x, 4 * cnum, 3, 1, name="pmconv10")
            pm = x
            x = tf.concat([x_hallu, pm], axis=3)

            x = gen_conv(x, 4 * cnum, 3, 1, name="allconv11")
            x = gen_conv(x, 4 * cnum, 3, 1, name="allconv12")
            x = gen_deconv(x, 2 * cnum, name="allconv13_upsample")
            x = gen_conv(x, 2 * cnum, 3, 1, name="allconv14")
            x = gen_deconv(x, cnum, name="allconv15_upsample")
            x = gen_conv(x, cnum // 2, 3, 1, name="allconv16")
            x = gen_conv(x, 4, 3, 1, activation=None, name="allconv17")
            x = tf.nn.tanh(x)
            x_stage2 = x
        return x_stage1, x_stage2, offset_flow

    def build_sn_patch_gan_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope("sn_patch_gan", reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, name="conv1", training=training)
            x = dis_conv(x, cnum * 2, name="conv2", training=training)
            x = dis_conv(x, cnum * 4, name="conv3", training=training)
            x = dis_conv(x, cnum * 4, name="conv4", training=training)
            x = dis_conv(x, cnum * 4, name="conv5", training=training)
            x = dis_conv(x, cnum * 4, name="conv6", training=training)
            x = flatten(x, name="flatten")
            return x

    def build_gan_discriminator(self, batch, reuse=False, training=True):
        with tf.variable_scope("discriminator", reuse=reuse):
            d = self.build_sn_patch_gan_discriminator(batch, reuse=reuse, training=training)
            return d

    def build_graph_with_losses(
        self,
        FLAGS,
        batch_data,
        batch_mask,
        batch_ref,
        identity_model,
        training=True,
        summary=False,
        reuse=False,
    ):

        if FLAGS.guided:
            batch_data, edge = batch_data
            edge = edge[:, :, :, 0:1] / 255.0
            edge = tf.cast(edge > FLAGS.edge_threshold, tf.float32)
        batch_pos = batch_data / 127.5 - 1.0
        if FLAGS.identity_loss:
            batch_ref_pos = batch_ref / 127.5 - 1.0
        else:
            batch_ref_pos = None

        # Load or generate mask, 1 represents masked point
        if batch_mask is None:
            bbox = random_bbox(FLAGS)
            regular_mask = bbox2mask(FLAGS, bbox, name="mask_c")
            irregular_mask = brush_stroke_mask(FLAGS, name="mask_c")
            mask = tf.cast(
                tf.logical_or(
                    tf.cast(irregular_mask, tf.bool),
                    tf.cast(regular_mask, tf.bool),
                ),
                tf.float32,
            )
        else:
            mask = batch_mask[:, :, :, -1]
            mask = tf.cast(mask > 0, tf.float32)
            mask = tf.expand_dims(mask, axis=-1)

        batch_incomplete = batch_pos * (1.0 - mask)
        if FLAGS.guided:
            edge = edge * mask
            xin = tf.concat([batch_incomplete, edge], axis=3)
        else:
            xin = batch_incomplete
        x1, x2, offset_flow = self.build_inpaint_net(
            xin,
            mask,
            batch_ref_pos,
            reuse=reuse,
            training=training,
            padding=FLAGS.padding,
            surface_attention=FLAGS.surface_attention,
        )
        batch_predicted = x2
        batch_predicted_x1 = x1
        losses = {}

        # Apply mask and complete image
        batch_complete = batch_predicted * mask + batch_incomplete * (1.0 - mask)
        batch_complete_x1 = batch_predicted_x1 * mask + batch_incomplete * (1.0 - mask)

        complete_x1_rgb = batch_complete_x1[:, :, :, :3]
        complete_x2_rgb = batch_complete[:, :, :, :3]
        complete_x1_depth_norm = tf.expand_dims(
            (batch_complete_x1[:, :, :, 3] + 1.0) * 127.5, axis=-1
        )
        complete_x2_depth_norm = tf.expand_dims(
            (batch_complete[:, :, :, 3] + 1.0) * 127.5, axis=-1
        )
        batch_pos_depth = (tf.expand_dims(batch_pos[:, :, :, 3], axis=-1) + 1.0) * 127.5
        ref_rgb = batch_ref_pos[:, :, :, :3]

        # Local patches
        losses["ae_loss"] = FLAGS.l1_loss_alpha * tf.reduce_mean(tf.abs(batch_pos - x1))
        losses["ae_loss"] += FLAGS.l1_loss_alpha * tf.reduce_mean(tf.abs(batch_pos - x2))

        sn_image_x2 = surface_conv(complete_x2_depth_norm)
        sn_image_pos = surface_conv(batch_pos_depth)
        if FLAGS.surface_loss:
            sn_image_x1 = surface_conv(complete_x1_depth_norm)
            losses["surface_loss"] = (
                tf.reduce_mean(tf.abs(sn_image_x1 - sn_image_pos)) * FLAGS.surface_loss_alpha
            )
            losses["surface_loss"] += (
                tf.reduce_mean(tf.abs(sn_image_x2 - sn_image_pos)) * FLAGS.surface_loss_alpha
            )

        if FLAGS.identity_loss:
            identity_loss_x1, identity_loss_x2 = gan_identity_loss(
                identity_model, complete_x1_rgb, complete_x2_rgb, ref_rgb
            )
            identity_loss_x1 *= FLAGS.identity_loss_alpha
            identity_loss_x2 *= FLAGS.identity_loss_alpha
            losses["identity_loss"] = identity_loss_x1 + identity_loss_x2

        if summary:
            scalar_summary("losses/ae_loss", losses["ae_loss"])
            if FLAGS.guided:
                viz_img = [batch_pos, batch_incomplete + edge, batch_complete]
            else:
                if FLAGS.identity_loss:
                    viz_img = [
                        batch_ref_pos,
                        batch_pos,
                        batch_incomplete,
                        batch_complete_x1,
                        batch_complete,
                    ]
                else:
                    viz_img = [batch_pos, batch_incomplete, batch_complete_x1, batch_complete]
            if offset_flow is not None:
                offset_resize = resize(offset_flow, scale=4, func=tf.image.resize_bilinear)
                a_channel = tf.ones(
                    [tf.shape(offset_resize)[0], FLAGS.img_shapes[0], FLAGS.img_shapes[1], 1]
                )
                offset_flow_a = tf.concat([a_channel, offset_resize], axis=3)
                viz_img.append(offset_flow_a)
            if not FLAGS.surface_loss:
                sn_image_x2 = surface_conv(complete_x2_depth_norm)
            sn_ones = tf.expand_dims(tf.ones(sn_image_x2.shape[0:3]), axis=-1)
            sn_image_x2_rgba = tf.concat([sn_image_x2, sn_ones], axis=3)
            viz_img.append(sn_image_x2_rgba)
            images_summary(
                tf.concat(viz_img, axis=2),
                "raw_incomplete_predicted_complete",
                FLAGS.viz_max_out,
                color_format="RGBA",
            )

        # GAN
        if FLAGS.surface_discriminator:
            batch_pos_sn = tf.concat([batch_pos, sn_image_pos], axis=-1)
            batch_complete_sn = tf.concat([batch_complete, sn_image_x2], axis=-1)
            batch_pos_neg = tf.concat([batch_pos_sn, batch_complete_sn], axis=0)
        else:
            batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)

        if FLAGS.gan_with_mask:
            if FLAGS.custom_mask:
                batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(mask, [2, 1, 1, 1])], axis=3)
            else:
                batch_pos_neg = tf.concat(
                    [batch_pos_neg, tf.tile(mask, [FLAGS.batch_size * 2, 1, 1, 1])], axis=3
                )

        if FLAGS.guided:
            # Conditional GANs
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(edge, [2, 1, 1, 1])], axis=3)

        # WGAN with gradient penalty
        if FLAGS.gan == "sngan":
            pos_neg = self.build_gan_discriminator(batch_pos_neg, training=training, reuse=reuse)
            pos, neg = tf.split(pos_neg, 2)
            g_loss, d_loss = gan_hinge_loss(pos, neg)
            losses["g_loss"] = g_loss
            losses["d_loss"] = d_loss
        else:
            raise NotImplementedError("{} not implemented.".format(FLAGS.gan))

        if summary:
            # Summarize the magnitude of gradients from different losses w.r.t. predicted image
            gradients_summary(losses["g_loss"], batch_predicted, name="g_loss")
            gradients_summary(losses["g_loss"], x2, name="g_loss_to_x2")
            gradients_summary(losses["ae_loss"], x1, name="ae_loss_to_x1")
            gradients_summary(losses["ae_loss"], x2, name="ae_loss_to_x2")

            if FLAGS.identity_loss:
                gradients_summary(losses["identity_loss"], x1, name="identity_loss_wrt_to_x1")
                gradients_summary(losses["identity_loss"], x2, name="identity_loss_wrt_to_x2")
            if FLAGS.surface_loss:
                gradients_summary(losses["surface_loss"], x1, name="surface_loss_wrt_to_x1")
                gradients_summary(losses["surface_loss"], x2, name="surface_loss_wrt_to_x2")
        losses["g_loss"] = FLAGS.gan_loss_alpha * losses["g_loss"]
        if FLAGS.ae_loss:
            losses["g_loss"] += losses["ae_loss"]
        if FLAGS.identity_loss:
            losses["g_loss"] += losses["identity_loss"]
        if FLAGS.surface_loss:
            losses["g_loss"] += losses["surface_loss"]
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inpaint_net")
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
        return g_vars, d_vars, losses

    def build_infer_graph(
        self, FLAGS, batch_data, batch_ref, batch_mask=None, bbox=None, name="val"
    ):
        """"""
        if FLAGS.guided:
            batch_data, edge = batch_data
            edge = edge[:, :, :, 0:1] / 255.0
            edge = tf.cast(edge > FLAGS.edge_threshold, tf.float32)

        if batch_mask is None:
            regular_mask = bbox2mask(FLAGS, bbox, name="mask_c")
            irregular_mask = brush_stroke_mask(FLAGS, name="mask_c")
            mask = tf.cast(
                tf.logical_or(
                    tf.cast(irregular_mask, tf.bool),
                    tf.cast(regular_mask, tf.bool),
                ),
                tf.float32,
            )
        else:
            mask = batch_mask[:, :, :, -1]
            mask = tf.cast(mask > 0, tf.float32)
            mask = tf.expand_dims(mask, axis=-1)

        if FLAGS.identity_loss:
            batch_ref = batch_ref / 127.5 - 1.0
        else:
            batch_ref = None
        batch_pos = batch_data / 127.5 - 1.0
        batch_incomplete = batch_pos * (1.0 - mask)
        if FLAGS.guided:
            edge = edge * mask
            xin = tf.concat([batch_incomplete, edge], axis=3)
        else:
            xin = batch_incomplete

        # Inpaint
        x1, x2, offset_flow = self.build_inpaint_net(
            xin,
            mask,
            batch_ref,
            reuse=True,
            training=False,
            padding=FLAGS.padding,
            surface_attention=FLAGS.surface_attention,
        )
        batch_predicted = x2
        batch_predicted_x1 = x1

        # Apply mask and reconstruct
        batch_complete = batch_predicted * mask + batch_incomplete * (1.0 - mask)
        batch_complete_x1 = batch_predicted_x1 * mask + batch_incomplete * (1.0 - mask)

        # Global image visualization
        if FLAGS.guided:
            viz_img = [batch_pos, batch_incomplete + edge, batch_complete]
        else:
            if FLAGS.identity_loss:
                viz_img = [
                    batch_ref,
                    batch_pos,
                    batch_incomplete,
                    batch_complete_x1,
                    batch_complete,
                ]
            else:
                viz_img = [batch_pos, batch_incomplete, batch_complete_x1, batch_complete]
        if offset_flow is not None:
            offset_resize = resize(offset_flow, scale=4, func=tf.image.resize_bilinear)
            a_channel = tf.ones(
                [tf.shape(offset_resize)[0], FLAGS.img_shapes[0], FLAGS.img_shapes[1], 1]
            )
            offset_flow_a = tf.concat([a_channel, offset_resize], axis=3)
            viz_img.append(offset_flow_a)
        complete_depth = tf.expand_dims(batch_complete[:, :, :, 3], axis=-1)
        depth_norm = (complete_depth + 1.0) * 127.5
        sn_image = surface_conv(depth_norm)
        sn_ones = tf.expand_dims(tf.ones(sn_image.shape[0:3]), axis=-1)
        sn_image_rgba = tf.concat([sn_image, sn_ones], axis=3)
        viz_img.append(sn_image_rgba)
        images_summary(
            tf.concat(viz_img, axis=2),
            name + "_raw_incomplete_complete",
            FLAGS.viz_max_out,
            color_format="RGBA",
        )
        return batch_complete

    def build_static_infer_graph(self, FLAGS, batch_data, batch_mask, batch_ref, name):
        """"""
        # Generate mask, 1 represents masked point
        if batch_mask == None:
            bbox = (
                tf.constant(FLAGS.height // 2),
                tf.constant(FLAGS.width // 2),
                tf.constant(FLAGS.height),
                tf.constant(FLAGS.width),
            )
        else:
            bbox = None
        return self.build_infer_graph(FLAGS, batch_data, batch_ref, batch_mask, bbox, name)

    def build_server_graph(self, FLAGS, batch_data, reuse=False, is_training=False):
        """"""
        # Generate mask, 1 represents masked point
        if FLAGS.guided:
            batch_raw, edge, masks_raw = tf.split(batch_data, 3, axis=2)
            edge = edge[:, :, :, 0:1] / 255.0
            edge = tf.cast(edge > FLAGS.edge_threshold, tf.float32)
        else:
            batch_raw, masks_raw, reference_raw = tf.split(batch_data, 3, axis=2)
        masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

        batch_pos = batch_raw / 127.5 - 1.0
        batch_ref_pos = reference_raw / 127.5 - 1.0
        batch_incomplete = batch_pos * (1.0 - masks)
        if FLAGS.guided:
            edge = edge * masks[:, :, :, 0:1]
            xin = tf.concat([batch_incomplete, edge], axis=3)
        else:
            xin = batch_incomplete
        # Inpaint
        x1, x2, flow = self.build_inpaint_net(
            xin,
            masks,
            batch_ref_pos,
            reuse=reuse,
            training=is_training,
            surface_attention=FLAGS.surface_attention,
        )
        batch_predict = x2
        batch_predict_x1 = x1
        # Apply mask and reconstruct
        batch_complete = batch_predict * masks + batch_incomplete * (1 - masks)
        batch_complete_x1 = batch_predict_x1 * masks + batch_incomplete * (1 - masks)
        # Construct surface normal image of result
        complete_depth = tf.expand_dims(batch_complete[:, :, :, 3], axis=-1)
        depth_norm = (complete_depth + 1.0) * 127.5
        sn_image = surface_conv(depth_norm)
        sn_ones = tf.expand_dims(tf.ones(sn_image.shape[0:3]), axis=-1)
        sn_image_rgba = tf.concat([sn_image, sn_ones], axis=3)
        viz_img = tf.concat(
            [
                batch_pos,
                batch_incomplete,
                batch_ref_pos,
                batch_complete_x1,
                batch_complete,
                sn_image_rgba,
            ],
            axis=2,
        )
        return batch_complete, viz_img
