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
import os, sys, yaml
import random
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from tf_transformer import Transformer
from tf_util import WarmupLRS




class Scholar():
    def __init__(self):
        parser = argparse.ArgumentParser(
        description='Scholar title prediction',
        usage='''cholar.py <command> [<args>]
Available Scholar commands are:
train       Train Scholar model
predict     Predict title from abstract
    ''')
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command', file=sys.stderr)
            parser.print_help(file=sys.stderr)
            exit(1)
        getattr(self, args.command)(sys.argv[2:])

    def train(self, argv):
        parser = argparse.ArgumentParser(description="Scholar training")
        parser.add_argument("records", help="Training records")
        parser.add_argument("vocabulary", help="Training vocabulary")
        parser.add_argument("--config", default=None, help="Transformer config file")
        parser.add_argument("--prefix", default="", help="Checkpoint and event prefix")
        parser.add_argument("--title_length", type=int, default=50, help="Maximum title length")
        parser.add_argument("--abstract_length", type=int, default=400, help="Maximum abstract length")
        parser.add_argument("--minibatch_size", type=int, default=32, help="Minibatch size")
        parser.add_argument("--batches_train", type=int, default=10000, help="Training batches")
        parser.add_argument("--batches_val", type=int, default=1000, help="Validation batches")
        parser.add_argument("--gpus", nargs='+', type=int, default=[], help="GPUs to use")
        args = parser.parse_args(argv)

        # input pipeline
        record_files = [os.path.join(dirpath, f) for dirpath, _, files
                            in os.walk(args.records) for f in files if f.endswith('.tfrec')]
        random.shuffle(record_files)
        val_rate = args.batches_train // args.batches_val
        val_split = int(max(1, args.batches_val / args.batches_train * len(record_files)))
        test_files = record_files[:val_split]
        train_files = record_files[val_split:]

        print("Training files {}".format(len(train_files)))
        print("Test files {}".format(len(test_files)))

        tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(args.vocabulary)

        def tf_parse(eg):
            example = tf.io.parse_example(
                eg[tf.newaxis], {
                    'title': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
                    'abstract': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
                    'journal' : tf.io.FixedLenFeature(shape=(), dtype=tf.string),
                    'year' : tf.io.FixedLenFeature(shape=(), dtype=tf.string)})
            abstract = example['abstract'][0]
            title = example['title'][0]
            return abstract, title

        def encode(abstract, title):
          abstract = [tokenizer_en.vocab_size] + tokenizer_en.encode(
              abstract.numpy()) + [tokenizer_en.vocab_size+1]
          title = [tokenizer_en.vocab_size] + tokenizer_en.encode(
              title.numpy()) + [tokenizer_en.vocab_size+1]
          return abstract, title

        def tf_encode(abstract, title):
            result_abstract, result_title = tf.py_function(encode, [abstract, title], [tf.int64, tf.int64])
            result_abstract.set_shape([None])
            result_title.set_shape([None])
            l0 = tf.cast(tf.expand_dims(tf.size(result_abstract), axis=-1) - 1, tf.int32)
            l1 = tf.cast(tf.expand_dims(tf.size(result_title), axis=-1) - 1, tf.int32)
            return ((result_abstract, l0), (result_title, l1))

        def tf_filter_max_length(x, y, abstract_length=args.abstract_length, title_length=args.title_length,):
            return tf.logical_and(x[1] <= abstract_length,
                                  y[1] <= title_length)[0]

        ds_train = tf.data.Dataset.from_tensor_slices(train_files)
        ds_train = (ds_train.interleave(lambda x:
                    tf.data.TFRecordDataset(filenames=x).map(tf_parse, num_parallel_calls=1), cycle_length=8, block_length=8))
        ds_train = (ds_train
                    .map(tf_encode)
                    .filter(tf_filter_max_length)
                    .prefetch(args.minibatch_size * 16)
                    .shuffle(args.minibatch_size * 16) # 1024
                    .padded_batch(args.minibatch_size,
                        padded_shapes=(([args.abstract_length,], [1,]), ([args.title_length,], [1,])),
                        drop_remainder=True)
                    .repeat())

        ds_test = tf.data.Dataset.from_tensor_slices(test_files)
        ds_test = (ds_test.interleave(lambda x:
                    tf.data.TFRecordDataset(filenames=x).map(tf_parse, num_parallel_calls=1), cycle_length=8, block_length=8))
        ds_test = (ds_test
                    .map(tf_encode)
                    .filter(tf_filter_max_length)
                    .prefetch(args.minibatch_size * 16)
                    .padded_batch(args.minibatch_size,
                        padded_shapes=(([args.abstract_length,], [1,]), ([args.title_length,], [1,])),
                        drop_remainder=True))

        transformer_hparams_file = (args.config or os.path.join('./training_configs', args.prefix, 'hparams.yaml'))
        if os.path.exists(transformer_hparams_file):
            with open(transformer_hparams_file, 'r') as fp:
                transformer_hparams = yaml.safe_load(fp)
        else:
            transformer_hparams = {
                               'd_output' : tokenizer_en.vocab_size + 2,
                               'd_model' : 512,
                               'dff' : 2048,
                               'nff' : 4,
                               'encoder_nff' : 2,
                               'decoder_nff' : 4,
                               #'dff_type' : 'point_wise' or 'convolution' or 'separable_convolution' or 'inception'
                               'encoder_dff_type' : 'point_wise',
                               'decoder_dff_type' : 'point_wise',
                               #'dff_filter_width', 'encoder_dff_filter_width', 'encoder_dff_pool_size'
                               # 'decoder_dff_filter_width', decoder_dff_pool_sizeh
                               'num_heads' : 8,
                               'encoder_max_iterations' : 14,   # 14
                               'decoder_max_iterations' : 14,
                               'encoder_time_scale' : 10000,
                               'decoder_time_scale' : 10000,
                               'ponder_bias_init' : 1.0,
                               #'act_type' : 'separable_convolution',
                               'encoder_act_type' : 'point_wise',
                               'decoder_act_type' : 'point_wise',
                               'act_dff' : None,
                               #'act_conv_filter' : 5,
                               'encoder_time_penalty' : 0.01,
                               'decoder_time_penalty' : 0.01,
                               }
            os.makedirs(os.path.dirname(transformer_hparams_file), exist_ok=True)
            with open(transformer_hparams_file, 'w') as fp:
                print(yaml.dump(transformer_hparams), file=fp)

        strategy = tf.distribute.MirroredStrategy(devices=['/gpu:' + str(i) for i in args.gpus] if args.gpus else ['/cpu:0'])

        ds_train_dist = strategy.experimental_distribute_dataset(ds_train)
        ds_test_dist = strategy.experimental_distribute_dataset(ds_test)

        print(next(iter(ds_train_dist)))

        checkpoint_dir = os.path.join('./training_checkpoints', args.prefix)
        os.makedirs(checkpoint_dir, exist_ok=True)
        summary_dir = os.path.join('./training_summaries', args.prefix)
        os.makedirs(summary_dir, exist_ok=True)
        summary_writer = tf.summary.create_file_writer(summary_dir)

        with strategy.scope(), summary_writer.as_default():
            tf_lrs = WarmupLRS(transformer_hparams.get('d_model'), warmup_steps=4000)
            tf_optimizer = tf.keras.optimizers.Adam(tf_lrs, beta_1=0.9, beta_2=0.98, epsilon=1e-9, amsgrad=False)
            cat_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')




if __name__ == '__main__':
    tf.config.optimizer.set_jit(True)
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)
    Scholar()
