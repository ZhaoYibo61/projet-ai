import tensorflow as tf
import time
import numpy as np
import tools as tools
import os as os

filenames = []
labels = []

filenames_test = []
labels_test = []

#https://github.com/wolfib/image-classification-CIFAR10-tf/blob/master/softmax.py
#https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428


beginTime = time.time()

# Parameter definitions
batch_size = 4680
learning_rate = 0.005
max_steps = 1000


def read_csv_on_folder(folder, name):
  import csv
  with open(folder + name, 'rt') as csvfile:
      reader = csv.reader(csvfile, delimiter=';', quotechar='|')
      next(reader)
      for row in reader:
        filenames.append(folder + row[0])
        labels.append(row[7])

def read_csv_test(folder, name):
  import csv
  with open(folder + name, 'rt') as csvfile:
      reader = csv.reader(csvfile, delimiter=';', quotechar='|')
      next(reader)
      for row in reader:
          filenames_test.append(folder + row[0])
          labels_test.append(row[7])



read_csv_on_folder("dataset/training/Final_Training/Images/00000/", "GT-00000.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00001/", "GT-00001.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00002/", "GT-00002.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00003/", "GT-00003.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00004/", "GT-00004.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00005/", "GT-00005.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00006/", "GT-00006.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00007/", "GT-00007.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00008/", "GT-00008.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00009/", "GT-00009.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00010/", "GT-00010.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00011/", "GT-00011.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00012/", "GT-00012.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00013/", "GT-00013.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00014/", "GT-00014.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00015/", "GT-00015.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00016/", "GT-00016.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00017/", "GT-00017.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00018/", "GT-00018.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00019/", "GT-00019.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00020/", "GT-00020.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00021/", "GT-00021.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00022/", "GT-00022.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00023/", "GT-00023.csv")
# read_csv_on_folder("dataset/training/Final_Training/Images/00024/", "GT-00024.csv")


read_csv_test("dataset/training/Test_Training/Images/", "GT-final_test.csv")






# Create a dataset returning slices of `filenames`
#raw_x_train = tf.data.Dataset.from_tensor(tffilenames_test)
#raw_y_train = tf.data.Dataset.from_tensor(tflabels_test)

# test_dataset = tf.data.Dataset.from_tensor_slices((tffilenames_test, tflabels_test))

# print(tffilenames)

# Parse every image in the dataset using `map`
def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=1)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])

    dict_image = tf.cast(image_resized, tf.float32)

    return dict_image

def _parse_function_labels(label):

    dict_label = tf.cast(label, tf.int32)
    return dict_label

# Batch size : taille de la fenetre


train_dataset_x = list(map(_parse_function, filenames_test))
train_dataset_y = list(map(_parse_function_labels, labels_test))



raw_x_train = tf.data.Dataset.from_tensor_slices(train_dataset_x)
raw_y_train = tf.data.Dataset.from_tensor_slices(train_dataset_y)

print(train_dataset_y.output_shapes)

"""
test_dataset_x = test_dataset.map(_parse_function)
test_dataset_y = test_dataset.map(_parse_function_labels)


def batch_generator(X, y, batch_size=64,
                    shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])

    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]

    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size, :], y[i:i + batch_size])




def conv_layer(input_tensor, name,
               kernel_size, n_output_channels,
               padding_mode='SAME', strides=(1, 1, 1, 1)):
    with tf.variable_scope(name):
        ## get n_input_channels:
        ##   input tensor shape:
        ##   [batch x width x height x channels_in]
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]

        weights_shape = (list(kernel_size) +
                         [n_input_channels, n_output_channels])

        weights = tf.get_variable(name='_weights',
                                  shape=weights_shape)
        print(weights)
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(
                                     shape=[n_output_channels]))
        print(biases)
        conv = tf.nn.conv2d(input=input_tensor,
                            filter=weights,
                            strides=strides,
                            padding=padding_mode)
        print(conv)
        conv = tf.nn.bias_add(conv, biases,
                              name='net_pre-activation')
        print(conv)
        conv = tf.nn.relu(conv, name='activation')
        print(conv)

        return conv


## testing
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    conv_layer(x, name='convtest', kernel_size=(3, 3), n_output_channels=32)

del g, x


# In[22]:


def fc_layer(input_tensor, name,
             n_output_units, activation_fn=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor,
                                      shape=(-1, n_input_units))

        weights_shape = [n_input_units, n_output_units]

        weights = tf.get_variable(name='_weights',
                                  shape=weights_shape)
        print(weights)
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(
                                     shape=[n_output_units]))
        print(biases)
        layer = tf.matmul(input_tensor, weights)
        print(layer)
        layer = tf.nn.bias_add(layer, biases,
                               name='net_pre-activation')
        print(layer)
        if activation_fn is None:
            return layer

        layer = activation_fn(layer, name='activation')
        print(layer)
        return layer


## testing:
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32,
                       shape=[None, 28, 28, 1])
    fc_layer(x, name='fctest', n_output_units=32,
             activation_fn=tf.nn.relu)

del g, x


# In[23]:


def build_cnn():
    ## Placeholders for X and y:
    tf_x = tf.placeholder(tf.float32, shape=[None, 784],
                          name='tf_x')
    tf_y = tf.placeholder(tf.int32, shape=[None],
                          name='tf_y')

    # reshape x to a 4D tensor:
    # [batchsize, width, height, 1]
    tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1],
                            name='tf_x_reshaped')
    ## One-hot encoding:
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=10,
                             dtype=tf.float32,
                             name='tf_y_onehot')

    ## 1st layer: Conv_1
    print('\nBuilding 1st layer: ')
    h1 = conv_layer(tf_x_image, name='conv_1',
                    kernel_size=(5, 5),
                    padding_mode='VALID',
                    n_output_channels=32)
    ## MaxPooling
    h1_pool = tf.nn.max_pool(h1,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
    ## 2n layer: Conv_2
    print('\nBuilding 2nd layer: ')
    h2 = conv_layer(h1_pool, name='conv_2',
                    kernel_size=(5, 5),
                    padding_mode='VALID',
                    n_output_channels=64)
    ## MaxPooling
    h2_pool = tf.nn.max_pool(h2,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')

    ## 3rd layer: Fully Connected
    print('\nBuilding 3rd layer:')
    h3 = fc_layer(h2_pool, name='fc_3',
                  n_output_units=1024,
                  activation_fn=tf.nn.relu)

    ## Dropout
    keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
    h3_drop = tf.nn.dropout(h3, keep_prob=keep_prob,
                            name='dropout_layer')

    ## 4th layer: Fully Connected (linear activation)
    print('\nBuilding 4th layer:')
    h4 = fc_layer(h3_drop, name='fc_4',
                  n_output_units=10,
                  activation_fn=None)

    ## Prediction
    predictions = {
        'probabilities': tf.nn.softmax(h4, name='probabilities'),
        'labels': tf.cast(tf.argmax(h4, axis=1), tf.int32,
                          name='labels')
    }

    ## Visualize the graph with TensorBoard:

    ## Loss Function and Optimization
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=h4, labels=tf_y_onehot),
        name='cross_entropy_loss')

    ## Optimizer:
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss,
                                   name='train_op')

    ## Computing the prediction accuracy
    correct_predictions = tf.equal(
        predictions['labels'],
        tf_y, name='correct_preds')

    accuracy = tf.reduce_mean(
        tf.cast(correct_predictions, tf.float32),
        name='accuracy')


def save(saver, sess, epoch, path='./model/'):
    if not os.path.isdir(path):
        os.makedirs(path)
    print('Saving model in %s' % path)
    saver.save(sess, os.path.join(path, 'cnn-model.ckpt'),
               global_step=epoch)


def load(saver, sess, path, epoch):
    print('Loading model from %s' % path)
    saver.restore(sess, os.path.join(
        path, 'cnn-model.ckpt-%d' % epoch))


def train(sess, training_set, validation_set=None,
          initialize=True, epochs=20, shuffle=True,
          dropout=0.5, random_seed=None):
    X_data = np.array(training_set[0])
    y_data = np.array(training_set[1])
    training_loss = []

    ## initialize variables
    if initialize:
        sess.run(tf.global_variables_initializer())

    np.random.seed(random_seed)  # for shuflling in batch_generator
    for epoch in range(1, epochs + 1):
        batch_gen = batch_generator(
            X_data, y_data,
            shuffle=shuffle)
        avg_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(batch_gen):
            feed = {'tf_x:0': batch_x,
                    'tf_y:0': batch_y,
                    'fc_keep_prob:0': dropout}
            loss, _ = sess.run(
                ['cross_entropy_loss:0', 'train_op'],
                feed_dict=feed)
            avg_loss += loss

        training_loss.append(avg_loss / (i + 1))
        print('Epoch %02d Training Avg. Loss: %7.3f' % (
            epoch, avg_loss), end=' ')
        if validation_set is not None:
            feed = {'tf_x:0': validation_set[0],
                    'tf_y:0': validation_set[1],
                    'fc_keep_prob:0': 1.0}
            valid_acc = sess.run('accuracy:0', feed_dict=feed)
            print(' Validation Acc: %7.3f' % valid_acc)
        else:
            print()


def predict(sess, X_test, return_proba=False):
    feed = {'tf_x:0': X_test,
            'fc_keep_prob:0': 1.0}
    if return_proba:
        return sess.run('probabilities:0', feed_dict=feed)
    else:
        return sess.run('labels:0', feed_dict=feed)


random_seed = 123

np.random.seed(random_seed)

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    ## build the graph
    build_cnn()

with tf.Session(graph=g) as sess:
    train(sess,
          training_set=(train_dataset_x, train_dataset_y),
          validation_set=(test_dataset_x, test_dataset_y),
          initialize=True,
          random_seed=123)

"""
