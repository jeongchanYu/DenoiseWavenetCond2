import tensorflow as tf
import json
import os
import custom_function as cf
import wav
import numpy as np
import denoise_wavenet_condition_test as DWC
import time
import datetime
import math
import make_dataset_test as md


# tf version check
tf_version = cf.get_tf_version()

# prevent GPU overflow
cf.tf_gpu_active_alloc()

# read config file
with open("config.json", "r") as f_json:
    config = json.load(f_json)

previous_size = config['previous_size']
frame_size = config["frame_size"]
future_size = config["future_size"]
shift_size = config["shift_size"]
window_type = config["window_type"]
dilation = config["dilation"]
num_of_speaker = config["num_of_speaker"]
num_of_noise = config["num_of_noise"]
if num_of_speaker < 2:
    num_of_speaker = 0
if num_of_noise < 2:
    num_of_noise = 0
max_condition = num_of_speaker + num_of_noise
if max_condition == 0:
    raise Exception("E: num_of_speaker and num_of_noise can't 0 same time in train_cond.py")

batch_size = 1
epochs = 1
learning_rate = config["learning_rate"]
default_float = config["default_float"]

train_source_path = config["train_source_path"]
train_target_path = config["train_target_path"]

load_check_point_name = config["load_check_point_name"]
save_check_point_name = config["save_check_point_name"]
save_check_point_period = config["save_check_point_period"]


# multi gpu init
# strategy = tf.distribute.experimental.CentralStorageStrategy()
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # make dataset
    train_source_cut_list, train_source_condition_list, train_target_cut_list, number_of_total_frame = md.make_dataset(train_source_path, train_target_path, batch_size,
                                                                 previous_size, frame_size, future_size, num_of_speaker,
                                                                 num_of_noise, shift_size, window_type, shuffle=True)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_source_cut_list, train_source_condition_list, train_target_cut_list)).shuffle(number_of_total_frame).batch(batch_size)
    dist_dataset_train = strategy.experimental_distribute_dataset(dataset=train_dataset)

    # make model
    model = DWC.DenoiseWavenetCondition(dilation, previous_size+frame_size+future_size, max_condition)
    loss_object = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    train_loss = tf.keras.metrics.Mean(name='train_loss')


# train function
@tf.function
def train_step(dist_inputs):
    def step_fn(inputs):
        x, cond, y = inputs
        cond = tf.reshape(cond, [-1, max_condition, 1])

        with tf.GradientTape() as tape:
            condition_list = model(cond)
        cond = tf.squeeze(cond)
        return cond, condition_list

    if tf_version[1] > 2:
        cond, condition_list = strategy.run(step_fn, args=(dist_inputs,))
    else:
        cond, condition_list = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    return cond, condition_list
# train run
with strategy.scope():
    # load model
    if load_check_point_name != "":
        saved_epoch = int(load_check_point_name.split('_')[-1])
        model.load_weights('{}/checkpoint/{}/data.ckpt'.format(cf.load_directory(), load_check_point_name))

    for epoch in range(saved_epoch, saved_epoch+epochs):
        i = 0
        for inputs in dist_dataset_train:
            cond, condition_list = train_step(inputs)
            print(cond.numpy(), np.array(condition_list))
            i += 1
