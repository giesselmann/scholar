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
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
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
        parser.add_argument("--title_length", type=int, default=25, help="Maximum title length")
        parser.add_argument("--abstract_length", type=int, default=300, help="Maximum abstract length")
        parser.add_argument("--minibatch_size", type=int, default=16, help="Minibatch size")
        parser.add_argument("--batches_train", type=int, default=3000000, help="Training batches")
        parser.add_argument("--batches_val", type=int, default=300000, help="Validation batches")
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
            return tf.logical_and(x[1] < abstract_length,
                                  y[1] < title_length)[0]

        ds_train = tf.data.Dataset.from_tensor_slices(train_files)
        ds_train = (ds_train.interleave(lambda x:
                    tf.data.TFRecordDataset(filenames=x).map(tf_parse, num_parallel_calls=4), cycle_length=8, block_length=8))
        ds_train = (ds_train
                    .map(tf_encode)
                    .filter(tf_filter_max_length)
                    .prefetch(args.minibatch_size * 64)
                    .shuffle(args.minibatch_size * 512) # 1024
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
                    .prefetch(args.minibatch_size * 64)
                    .padded_batch(args.minibatch_size,
                        padded_shapes=(([args.abstract_length,], [1,]), ([args.title_length,], [1,])),
                        drop_remainder=True)
                    .repeat())

        transformer_hparams_file = (args.config or os.path.join('./training_configs', args.prefix, 'hparams.yaml'))
        if os.path.exists(transformer_hparams_file):
            with open(transformer_hparams_file, 'r') as fp:
                transformer_hparams = yaml.safe_load(fp)
        else:
            transformer_hparams = {
                               'd_output' : tokenizer_en.vocab_size + 2,
                               'd_model' : 512,
                               'dff' : 2048,
                               #'nff' : 4,
                               'encoder_nff' : 3,
                               'decoder_nff' : 3,
                               #'dff_type' : 'point_wise' or 'convolution' or 'separable_convolution' or 'inception'
                               'encoder_dff_type' : 'point_wise',
                               'decoder_dff_type' : 'point_wise',
                               #'dff_filter_width', 'encoder_dff_filter_width', 'encoder_dff_pool_size'
                               # 'decoder_dff_filter_width', decoder_dff_pool_sizeh
                               'num_heads' : 8,
                               'encoder_max_iterations' : 8,   # 14
                               'decoder_max_iterations' : 8,
                               'encoder_time_scale' : 10000,
                               'decoder_time_scale' : 10000,
                               'ponder_bias_init' : 1.0,
                               #'act_type' : 'separable_convolution',
                               'encoder_act_type' : 'point_wise',
                               'decoder_act_type' : 'point_wise',
                               'act_dff' : None,
                               #'act_conv_filter' : 5,
                               'encoder_time_penalty' : 0.05,
                               'decoder_time_penalty' : 0.05,
                               }
            os.makedirs(os.path.dirname(transformer_hparams_file), exist_ok=True)
            with open(transformer_hparams_file, 'w') as fp:
                print(yaml.dump(transformer_hparams), file=fp)

        strategy = tf.distribute.MirroredStrategy(devices=['/gpu:' + str(i) for i in args.gpus] if args.gpus else ['/cpu:0'])

        ds_train_dist = strategy.experimental_distribute_dataset(ds_train)
        ds_test_dist = strategy.experimental_distribute_dataset(ds_test)

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
            # Transformer loss function
            def tf_loss_function(real, pred, mask):
                mask = mask / tf.reduce_sum(mask, axis=-1, keepdims=True)
                #mask /= tf.cast(tf.size(mask), tf.float32)
                loss_ = cat_cross_entropy(real, pred, sample_weight=mask) # (batch_size, target_seq_len)
                return tf.nn.compute_average_loss(loss_, global_batch_size=args.minibatch_size)

            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

            transformer = Transformer(hparams=transformer_hparams, name='Transformer')

            tf_checkpoint = tf.train.Checkpoint(optimizer=tf_optimizer, transformer=transformer)
            tf_ckpt_manager = tf.train.CheckpointManager(tf_checkpoint, checkpoint_dir, max_to_keep=5)

            if tf_ckpt_manager.latest_checkpoint:
                tf_checkpoint.restore(tf_ckpt_manager.latest_checkpoint)
                print('Latest transformer checkpoint restored!!')

            def train_step(inputs):
                (input_data, input_lengths), (target_data, target_lengths) = inputs
                mask = tf.sequence_mask(tf.squeeze(target_lengths), target_data.shape[1] - 1, dtype=tf.float32)
                # Transformer gradient
                with tf.GradientTape() as tape:
                    tf_predictions, _enc_loss, _dec_loss = transformer([input_data, input_lengths, target_data[:,:-1], target_lengths], training=True)
                    tf_loss = tf_loss_function(target_data[:,1:], tf_predictions, mask)
                    enc_loss = tf.nn.compute_average_loss(_enc_loss, global_batch_size=args.minibatch_size)
                    dec_loss = tf.nn.compute_average_loss(_dec_loss, global_batch_size=args.minibatch_size)
                tf_gradients = tape.gradient([tf_loss, enc_loss, dec_loss], transformer.trainable_variables)
                tf_gradients, _ = tf.clip_by_global_norm(tf_gradients, 10.0)
                # Apply gradients
                tf_optimizer.apply_gradients(zip(tf_gradients, transformer.trainable_variables))
                # reset and update accuracies
                train_accuracy.update_state(target_data[:,1:], tf_predictions, mask)
                return tf_loss

            def test_step(inputs):
                (input_data, input_lengths), (target_data, target_lengths) = inputs
                mask = tf.sequence_mask(tf.squeeze(target_lengths),  target_data.shape[1] - 1, dtype=tf.float32)
                tf_predictions, _, _ = transformer([input_data, input_lengths, target_data[:,:-1], target_lengths], training=False)
                t_loss = tf_loss_function(target_data[:,1:], tf_predictions, mask)
                test_accuracy.update_state(target_data[:,1:], tf_predictions, mask)
                return t_loss

            # `experimental_run_v2` replicates the provided computation and runs it
            # with the distributed input.
            @tf.function
            def distributed_train_step(dataset_inputs):
                per_replica_tf_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
                return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_tf_losses, axis=None)

            @tf.function
            def distributed_test_step(dataset_inputs):
                per_replica_tf_losses = strategy.experimental_run_v2(test_step, args=(dataset_inputs,))
                return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_tf_losses, axis=None)

            best_losses = [float("inf")] * 5
            min_steps = 250
            steps = 0
            for epoch in range(20):
                total_loss = 0.0
                num_batches = 0
                ds_train_dist_iter = iter(ds_train_dist)
                ds_test_dist_iter = iter(ds_test_dist)
                for batch in tqdm(range(args.batches_train), desc='Training', ncols=0):
                    tf.summary.experimental.set_step(tf_optimizer.iterations)
                    batch_input = next(ds_train_dist_iter)
                    tf_loss = distributed_train_step(batch_input)
                    steps += 1
                    if epoch == 0 and batch == 0:
                        transformer.summary()
                    if tf_loss <= max(best_losses) and steps > min_steps:
                        tf_ckpt_manager.save()
                        best_losses[best_losses.index(max(best_losses))] = tf_loss
                        steps = 0
                    num_batches += 1
                    total_loss += tf_loss
                    train_loss = total_loss / num_batches
                    tf.summary.scalar("loss", tf_loss)
                    tf.summary.scalar("lr", tf_lrs(tf_optimizer.iterations.numpy().astype(np.float32)))
                    if batch % val_rate == 0:
                        batch_input = next(ds_test_dist_iter)
                        test_loss = distributed_test_step(batch_input)
                        tf.summary.scalar("loss_val", test_loss)
                        tf.summary.scalar('acc_train', train_accuracy.result())
                        tf.summary.scalar("acc_val", test_accuracy.result())
                        train_accuracy.reset_states()
                        test_accuracy.reset_states()
                print("Epoch {}: train loss: {}".format(epoch, train_loss))

    def predict(self, argv):
        pass




if __name__ == '__main__':
    tf.config.optimizer.set_jit(True)
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)
    Scholar()
