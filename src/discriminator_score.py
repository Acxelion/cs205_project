#!/usr/bin/env python3
"""This script scores pianoroll inputs using a trained discriminator."""
import os
import logging
import argparse
from pprint import pformat
import numpy as np
import tensorflow as tf
from musegan.config import LOGLEVEL, LOG_FORMAT
from musegan.data import load_data
from musegan.model import Model
from musegan.utils import make_sure_path_exists, load_yaml, update_not_none, load_component

LOGGER = logging.getLogger("musegan.discriminator_score")

def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Score pianoroll data using a trained discriminator."
    )
    parser.add_argument('--result_dir',
                        help="Directory where the results are saved.")
    parser.add_argument('--checkpoint_dir',
                        required=True,
                        help="Directory that contains checkpoints.")
    parser.add_argument('--params', '--params_file', '--params_file_path',
                        required=True,
                        help="Path to the file that defines the hyperparameters.")
    parser.add_argument('--config',
                        help="Path to the configuration file.")
    parser.add_argument('--input_npy',
                        required=True,
                        help="Path to a .npy file of shape [batch] + data_shape.")
    parser.add_argument('--output_raw', default='raw_logits.npy',
                        help="Path to save raw discriminator logits.")
    parser.add_argument('--output_prob', default='probs.npy',
                        help="Path to save sigmoid probabilities.")
    parser.add_argument('--output_aggregated', default='aggregated_probs.npy',
                        help="Path to save aggregated (per-sample) sigmoid probabilities.")
    parser.add_argument('--output_aggregated_raw', default='aggregated_raw.npy',
                        help="Path to save aggregated (per-sample) raw logits.")
    parser.add_argument('--aggregation_method', default='mean', choices=['mean', 'min', 'max'],
                        help="Method to aggregate multiple scores per sample (mean, min, max).")
    parser.add_argument('--gpu', '--gpu_device_num', type=str, default="0",
                        help="The GPU device number to use.")
    args = parser.parse_args()
    return args

def setup():
    """Parse command line arguments, load model parameters, load configurations
    and setup environment."""
    # Parse the command line arguments
    args = parse_arguments()

    # Load parameters
    params = load_yaml(args.params)

    # Load configurations
    config = {}
    if args.config:
        config = load_yaml(args.config)
    
    # Update config with command-line args
    update_not_none(config, vars(args))

    # Make sure result directory exists if specified
    if 'result_dir' in config and config['result_dir']:
        make_sure_path_exists(config['result_dir'])

    # Setup GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']

    return params, config

def main():
    """Main function."""
    # Setup
    logging.basicConfig(level=LOGLEVEL, format=LOG_FORMAT)
    params, config = setup()
    LOGGER.info("Using parameters:\n%s", pformat(params))
    LOGGER.info("Using config:\n%s", pformat(config))

    # Placeholder for input data
    x_ph = tf.placeholder(tf.float32, [None] + params['data_shape'], name='x_input')

    # Build discriminator graph
    DiscClass = load_component('discriminator',
                               params['nets']['discriminator'],
                               'Discriminator')
    disc = DiscClass(n_tracks=params['data_shape'][-1],
                    beat_resolution=params.get('beat_resolution'))
    with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
        logits = disc(x_ph, training=False)
    probs = tf.sigmoid(logits)

    # Get tensorflow session config
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # Create saver to restore variables
    saver = tf.train.Saver()

    # Tensorflow Session
    with tf.Session(config=tf_config) as sess:
        # Restore the latest checkpoint
        LOGGER.info("Restoring the latest checkpoint.")
        with open(os.path.join(config['checkpoint_dir'], 'checkpoint')) as f:
            checkpoint_name = os.path.basename(
                f.readline().split()[1].strip('"'))
        checkpoint_path = os.path.realpath(
            os.path.join(config['checkpoint_dir'], checkpoint_name))
        saver.restore(sess, checkpoint_path)
        
        # Load and score inputs
        LOGGER.info("Loading input data from %s", config['input_npy'])
        X = np.load(config['input_npy'])
        batch_size = X.shape[0]
        LOGGER.info("Scoring %d samples", batch_size)
        raw_scores, prob_scores = sess.run([logits, probs], feed_dict={x_ph: X})

    # Save original outputs
    LOGGER.info("Saving results...")
    np.save(config['output_raw'], raw_scores)
    np.save(config['output_prob'], prob_scores)
    LOGGER.info("Saved raw logits to %s", config['output_raw'])
    LOGGER.info("Saved probabilities to %s", config['output_prob'])
    
    # Reshape and aggregate scores (each sample has multiple scores)
    window_count = raw_scores.shape[0] // batch_size
    LOGGER.info("Detected %d windows per sample", window_count)
    
    reshaped_probs = prob_scores.reshape(batch_size, window_count)
    reshaped_raw = raw_scores.reshape(batch_size, window_count)
    
    # Aggregate scores based on specified method
    if config.get('aggregation_method') == 'min':
        aggregated_scores = np.min(reshaped_probs, axis=1, keepdims=True)
        aggregated_raw = np.min(reshaped_raw, axis=1, keepdims=True)
        LOGGER.info("Using MIN aggregation method")
    elif config.get('aggregation_method') == 'max':
        aggregated_scores = np.max(reshaped_probs, axis=1, keepdims=True)
        aggregated_raw = np.max(reshaped_raw, axis=1, keepdims=True)
        LOGGER.info("Using MAX aggregation method")
    else:  # default to mean
        aggregated_scores = np.mean(reshaped_probs, axis=1, keepdims=True)
        aggregated_raw = np.mean(reshaped_raw, axis=1, keepdims=True)
        LOGGER.info("Using MEAN aggregation method")
    
    # Save aggregated scores
    np.save(config['output_aggregated'], aggregated_scores)
    np.save(config['output_aggregated_raw'], aggregated_raw)
    LOGGER.info("Saved aggregated probabilities to %s", config['output_aggregated'])
    LOGGER.info("Saved aggregated raw logits to %s", config['output_aggregated_raw'])

if __name__ == "__main__":
    main() 