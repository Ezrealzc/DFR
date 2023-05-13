import tensorflow as tf
from keras.layers import Input, Lambda, MaxPooling2D
from keras.models import Model
from nets.DFRnet_training import loss
from nets.feature import HFF,DFR_head


def nms(heat, kernel=3):
    hmax = MaxPooling2D((kernel, kernel), strides=1, padding='SAME')(heat)
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat

def topk(hm, max_objects=1000):
    hm = nms(hm)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    hm = tf.reshape(hm, (b, -1))
    scores, indices = tf.math.top_k(hm, k=max_objects, sorted=True)

    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys

def decode(hm, wh, reg, max_objects=1000):
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    reg     = tf.reshape(reg, [b, -1, 2])
    wh      = tf.reshape(wh, [b, -1, 2])
    length  = tf.shape(wh)[1]
    batch_idx       = tf.expand_dims(tf.range(0, b), 1)
    batch_idx       = tf.tile(batch_idx, (1, max_objects))
    full_indices    = tf.reshape(batch_idx, [-1]) * tf.to_int32(length) + tf.reshape(indices, [-1])
    topk_reg = tf.gather(tf.reshape(reg, [-1, 2]), full_indices)
    topk_reg = tf.reshape(topk_reg, [b, -1, 2])
    topk_wh = tf.gather(tf.reshape(wh, [-1, 2]), full_indices)
    topk_wh = tf.reshape(topk_wh, [b, -1, 2])

    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]

    topk_x1, topk_y1 = topk_cx - topk_wh[..., 0:1] / 2, topk_cy - topk_wh[..., 1:2] / 2
    topk_x2, topk_y2 = topk_cx + topk_wh[..., 0:1] / 2, topk_cy + topk_wh[..., 1:2] / 2
    
    scores      = tf.expand_dims(scores, axis=-1)
    class_ids   = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)

    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)

    return detections

def DFR(input_shape, num_classes, backbone='HFF', max_objects=1000, mode="train", num_stacks=2):

    image_input     = Input(shape=input_shape)

    if backbone=='HFF':
        #-----------------------------------#
        #   对输入图片进行特征提取
        #-----------------------------------#
        fuse_feature = HFF(image_input)
        y1, y2, y3 = DFR_head(fuse_feature, num_classes)

        if mode=="train":
            model = Model(inputs=image_input, outputs=[y1, y2, y3])
            detections = Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            prediction_model    = Model(inputs=image_input, outputs=detections)
            return model, prediction_model
        elif mode=="predict":
            detections = Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            prediction_model = Model(inputs=image_input, outputs=detections)
            return prediction_model
        elif mode=="heatmap":
            prediction_model = Model(inputs=image_input, outputs=y1)
            return prediction_model


def get_train_model(model_body, input_shape, num_classes, backbone='HFF', max_objects=200):
    output_size     = input_shape[0] // 4
    hm_input        = Input(shape=(output_size, output_size, num_classes))
    wh_input        = Input(shape=(max_objects, 2))
    reg_input       = Input(shape=(max_objects, 2))
    reg_mask_input  = Input(shape=(max_objects,))
    index_input     = Input(shape=(max_objects,))

    if backbone=='HFF':
        y1, y2, y3 = model_body.output
        loss_ = Lambda(loss, output_shape = (1, ),name='DFR_loss')([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
        model = Model(inputs=[model_body.input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])
    else:
        outs = model_body.output

        loss_all = []
        for i in range(len(outs) // 3):  
            y1, y2, y3 = outs[0 + i * 3], outs[1 + i * 3], outs[2 + i * 3]
            loss_ = Lambda(loss)([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
            loss_all.append(loss_)
        loss_all =  Lambda(tf.reduce_mean, output_shape = (1, ),name='DFR_loss')(loss_)

        model = Model(inputs=[model_body.input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_all])
    return model
