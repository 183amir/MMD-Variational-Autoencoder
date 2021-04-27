import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
from matplotlib import pyplot as plt
import math, os
# from tensorflow.examples.tutorials.mnist import input_data

# Define some handy network layers
def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)


def conv2d_lrelu(inputs, num_outputs, kernel_size, stride):
    conv = tf.compat.v1.layers.conv2d(
        inputs,
        num_outputs,
        kernel_size,
        stride,
        padding="SAME",
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        ),
        activation=tf.identity,
    )
    conv = lrelu(conv)
    return conv


def conv2d_t_relu(inputs, num_outputs, kernel_size, stride):
    conv = tf.compat.v1.layers.conv2d_transpose(
        inputs,
        num_outputs,
        kernel_size,
        strides=stride,
        padding="SAME",
        activation=tf.identity,
        use_bias=True,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        ),
    )
    conv = tf.nn.relu(conv)
    return conv


def fc_lrelu(inputs, num_outputs):
    fc = tf.compat.v1.layers.dense(
        inputs,
        num_outputs,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        ),
        activation=tf.identity,
    )
    fc = lrelu(fc)
    return fc


def fc_relu(inputs, num_outputs):
    fc = tf.compat.v1.layers.dense(
        inputs,
        num_outputs,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        ),
        activation=tf.identity,
    )
    fc = tf.nn.relu(fc)
    return fc


# Encoder and decoder use the DC-GAN architecture
def encoder(x, z_dim):
    with tf.compat.v1.variable_scope("encoder"):
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        return tf.compat.v1.layers.dense(fc1, z_dim, activation=tf.identity)


def decoder(z, reuse=False):
    with tf.compat.v1.variable_scope("decoder") as vs:
        if reuse:
            vs.reuse_variables()
        fc1 = fc_relu(z, 1024)
        fc2 = fc_relu(fc1, 7 * 7 * 128)
        fc2 = tf.reshape(fc2, tf.stack([tf.shape(input=fc2)[0], 7, 7, 128]))
        conv1 = conv2d_t_relu(fc2, 64, 4, 2)
        output = tf.compat.v1.layers.conv2d_transpose(
            conv1, 1, 4, 2, activation=tf.sigmoid, padding="same"
        )
        return output


def compute_kernel(x, y):
    x_size = tf.shape(input=x)[0]
    y_size = tf.shape(input=y)[0]
    dim = tf.shape(input=x)[1]
    tiled_x = tf.tile(
        tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1])
    )
    tiled_y = tf.tile(
        tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1])
    )
    return tf.exp(
        -tf.reduce_mean(input_tensor=tf.square(tiled_x - tiled_y), axis=2)
        / tf.cast(dim, tf.float32)
    )


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return (
        tf.reduce_mean(input_tensor=x_kernel)
        + tf.reduce_mean(input_tensor=y_kernel)
        - 2 * tf.reduce_mean(input_tensor=xy_kernel)
    )


# Convert a numpy array of shape [batch_size, height, width, 1] into a displayable array
# of shape [height*sqrt(batch_size, width*sqrt(batch_size))] by tiling the images
def convert_to_display(samples):
    cnt, height, width = (
        int(math.floor(math.sqrt(samples.shape[0]))),
        samples.shape[1],
        samples.shape[2],
    )
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width])
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height * cnt, width * cnt])
    return samples


batch_size = 200
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images /= 255.
train_dataset = tf.compat.v1.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(batch_size).repeat()

# plt.ion()

# Build the computation graph for training
z_dim = 20
train_x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1])
train_z = encoder(train_x, z_dim)
train_xr = decoder(train_z)

# Build the computation graph for generating samples
gen_z = tf.compat.v1.placeholder(tf.float32, shape=[None, z_dim])
gen_x = decoder(gen_z, reuse=True)

# Compare the generated z with true samples from a standard Gaussian, and compute their MMD distance
true_samples = tf.random.normal(tf.stack([200, z_dim]))
loss_mmd = compute_mmd(true_samples, train_z)
loss_nll = tf.reduce_mean(input_tensor=tf.square(train_xr - train_x))
loss = loss_nll + loss_mmd
trainer = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# Start training
next_batch = train_dataset.make_one_shot_iterator().get_next()

for i in range(10001):
# for i, batch_x in enumerate(train_dataset):
#     if i == 10001:
#         break
    # batch_x, batch_y = mnist.train.next_batch(batch_size)
    # batch_x = batch_x.reshape(-1, 28, 28, 1)
    batch_x = sess.run(next_batch)
    _, nll, mmd = sess.run([trainer, loss_nll, loss_mmd], feed_dict={train_x: batch_x})
    if i % 100 == 0:
        print("Negative log likelihood is %f, mmd loss is %f" % (nll, mmd))
    if i % 500 == 0:
        samples = sess.run(
            gen_x, feed_dict={gen_z: np.random.normal(size=(100, z_dim))}
        )
        plt.imshow(convert_to_display(samples), cmap="Greys_r")
        plt.savefig(f"samples_{i:00d}.png")
        # plt.show()
        # plt.pause(0.001)

# If latent z is 2-dimensional we visualize it by plotting latent z of different digits in different colors
if z_dim == 2:
    z_list, label_list = [], []
    test_batch_size = 500
    for i in range(20):
        batch_x = sess.run(next_batch)
    #     batch_x, batch_y = mnist.test.next_batch(test_batch_size)
    #     batch_x = batch_x.reshape(-1, 28, 28, 1)
    # for i, batch_x in enumerate(train_dataset):
    #     if i == 20:
    #         break
        z_list.append(sess.run(train_z, feed_dict={train_x: batch_x}))
        # label_list.append(batch_y)
    z = np.concatenate(z_list, axis=0)
    # label = np.concatenate(label_list)
    plt.scatter(z[:, 0], z[:, 1])#, c=label)
    plt.savefig(f"scatter.png")
