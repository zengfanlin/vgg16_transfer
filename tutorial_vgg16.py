from recordutil import *
from vgg16model import *

n_epoch = 200
learning_rate = 0.0001
print_freq = 2
batch_size = 20


def fc_layers(net):
    # 预处理
    network = FlattenLayer(net, name='flatten')
    network = DenseLayer(network, n_units=256, act=tf.nn.relu, name='fc6')
    # network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc2_relu')
    network = DenseLayer(network, n_units=2, act=tf.identity, name='out')
    return network


sess = tf.InteractiveSession()
# 输入
x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x')
# 输出
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

net_in = InputLayer(x, name='input')
# net_cnn = conv_layers(net_in)               # professional CNN APIs
net_cnn = conv_layers_simple_api(net_in)  # simplified CNN APIs
network = fc_layers(net_cnn)

y = network.outputs
# probs = tf.nn.softmax(y)
y_op = tf.argmax(tf.nn.softmax(y), 1)
cost = tl.cost.cross_entropy(y, y_, name='cost')
correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.float32), tf.cast(y_, tf.float32))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 定义 optimizer
train_params = network.all_params[26:]
# print(train_params)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
                                  epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

img, label = read_and_decode("F:\\001-Python\\train.tfrecords")
# img_v, label_v = read_and_decode("F:\\001-Python\\val.tfrecords")

# 使用shuffle_batch可以随机打乱输入
X_train, y_train = tf.train.shuffle_batch([img, label],
                                          batch_size=batch_size, capacity=600,
                                          min_after_dequeue=500)

# X_Val, y_val = tf.train.shuffle_batch([img_v, label_v],
#                                       batch_size=30, capacity=400,
#                                       min_after_dequeue=300)

tl.layers.initialize_global_variables(sess)

network.print_params()
network.print_layers()

npz = np.load('vgg16_weights.npz')

params = []

for val in sorted(npz.items())[0:26]:
    print("  Loading %s" % str(val[0]))
    print("  Loading %s" % str(val[1].shape))
    params.append(val[1])
# 加载预训练的参数
tl.files.assign_params(sess, params, network)

# tl.files.load_and_assign_npz(sess=sess, name='model2.npz', network=network)


# 训练模型

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
step = 0

for epoch in range(n_epoch):
    start_time = time.time()
    val, l = sess.run([X_train, y_train])
    # val_, l_ = sess.run([X_Val, y_val])
    # tl.utils.fit(sess, network, train_op, cost, val, l, x, y_,
    #              acc=acc, batch_size=15, n_epoch=1, print_freq=2,
    #              X_val=None, y_val=None, eval_train=False)

    for X_train_a, y_train_a in tl.iterate.minibatches(val, l, batch_size, shuffle=True):
        sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

    if epoch + 1 == 1 or (epoch + 1) % 5 == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(val, l, batch_size, shuffle=True):
            err, ac = sess.run([cost, acc], feed_dict={x: X_train_a, y_: y_train_a})
            train_loss += err
            train_acc += ac
            n_batch += 1
        print("   train loss: %f" % (train_loss / n_batch))
        print("   train acc: %f" % (train_acc / n_batch))
        tl.files.save_npz(network.all_params, name='model3.npz', sess=sess)
coord.request_stop()
coord.join(threads)

sess.close()
