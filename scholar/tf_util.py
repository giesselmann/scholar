# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : Scholar
#
#  DESCRIPTION   : Title from abstract prediction
#
#  RESTRICTIONS  : none
#
#  REQUIRES      : none
#
# ---------------------------------------------------------------------------------
# Copyright 2019-2020 Pay Giesselmann, Max Planck Institute for Molecular Genetics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Pay Giesselmann
# ---------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf





def positional_encoding(seq_len, depth, d_model,
                        max_timescale=10000, random_shift=False):
    def get_angles(pos, j, max_timescale):
        angle_rates = 1 / np.power(max_timescale,
                                   (2 * (j//2)) / np.float32(d_model))
        return pos * angle_rates
    j = np.arange(d_model)[np.newaxis, np.newaxis, :]   # (1, 1, d_model)
    pos_rads = get_angles(np.arange(seq_len)[np.newaxis, :, np.newaxis], j, max_timescale)
    #depth_rads = get_angles(np.arange(depth)[:, np.newaxis, np.newaxis],
    #                        np.arange(d_model)[np.newaxis, np.newaxis, :])
    #depth_rads = get_angles(np.logspace(0, np.log10(max_timescale), depth)[:, np.newaxis, np.newaxis], j)
    depth_rads = get_angles(np.arange(depth)[:, np.newaxis, np.newaxis], j, depth)
    rads = pos_rads + depth_rads
    # apply sin to even indices in the array; 2i
    rads[:, 0::2, :] = np.sin(pos_rads[:, 0::2, :]) + np.sin(depth_rads[:, :, :])
    # apply cos to odd indices in the array; 2i+1
    rads[:, 1::2, :] = np.cos(pos_rads[:, 1::2, :]) + np.cos(depth_rads[:, :, :])
    pos_encoding = rads[np.newaxis, ...]
    return tf.constant(tf.cast(pos_encoding, dtype=tf.float32)) # (1, depth, seq_len, d_model)




class WarmupLRS(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, offset=0):
        super(WarmupLRS, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
        self.offset = offset

    def update_offset(self, update):
        self.offset += update

    def __call__(self, step):
        step = tf.maximum(tf.cast(0, step.dtype), step - tf.cast(self.offset, step.dtype))
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
