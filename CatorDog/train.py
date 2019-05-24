import tensorflow as tf
import dataset

batch_size = 32

# 分类类别
classes = ['dogs','cats']
num_classes=len(classes)

# 交叉验证集比例
# 输入图片大小
# 输入图片通道数
# 训练数据路径
validation_rate=0.25
img_size=64
num_channels=3
train_data_path='training_data'

# 读数据，预处理
data=dataset.read_data(train_data_path,img_size,classes,validation_rate)

# print(format(len(data.train.labels)))

# 设定参数，构建网络

# 设定占位符
x=tf.placeholder(tf.float32,shape=[None,img_size,img_size,num_channels],name='x')
y_true=tf.placeholder(tf.float32,shape=[None,num_classes],name='y_true')
y_true_cls=tf.argmax(y_true,dimension=1)
# 网络结构参数
# 第一层
# 卷积

w_layer1_conv=tf.Variable(tf.truncated_normal([3, 3, 3, 16], stddev=0.05))
b_layer1_conv=tf.Variable(tf.constant(0.05, shape=[16]))

layer1_conv= tf.nn.conv2d(x, w_layer1_conv, strides=[1, 1, 1, 1], padding='SAME')
layer1_conv += b_layer1_conv

layer1_conv=tf.nn.relu(layer1_conv)

# 池化
layer1_pooling=tf.nn.max_pool(layer1_conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 第二层
# 卷积
w_layer2_conv=tf.Variable(tf.truncated_normal([3,3,16,32],stddev=0.05))
b_layer2_conv=tf.Variable(tf.constant(0.05,shape=[32]))

layer2_conv=tf.nn.conv2d(layer1_pooling,w_layer2_conv,strides=[1,1,1,1],padding='SAME')
layer2_conv +=b_layer2_conv

layer2_conv=tf.nn.relu(layer2_conv)

# 池化
layer2_pooling=tf.nn.max_pool(layer2_conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 第三层
# 卷积
w_layer3_conv=tf.Variable(tf.truncated_normal([3,3,32,64],stddev=0.05))
b_layer3_conv=tf.Variable(tf.constant(0.05,shape=[64]))

layer3_conv=tf.nn.conv2d(layer2_pooling,w_layer3_conv,strides=[1,1,1,1],padding='SAME')
layer3_conv += b_layer3_conv

layer3_conv=tf.nn.relu(layer3_conv)

# 池化
layer3_pooling=tf.nn.max_pool(layer3_conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 第四层平滑层,主要是将三维拉成一维
#None 8,8,64 到4096
layer4_shape=layer3_pooling.get_shape()
num_features=layer4_shape[1:4].num_elements()
layer4_flatten=tf.reshape(layer3_pooling,[-1,num_features])

# 第五层全连接
w_layer5_fc1=tf.Variable(tf.truncated_normal([num_features,1024],stddev=0.05))
b_layer5_fc1=tf.Variable(tf.constant(0.05,shape=[1024]))

layer5_fc1=tf.matmul(layer4_flatten,w_layer5_fc1)+b_layer5_fc1

layer5_fc1=tf.nn.relu(tf.nn.dropout(layer5_fc1,keep_prob=0.7))


# 第六层全连接
w_layer6_fc2=tf.Variable(tf.truncated_normal([1024,2],stddev=0.05))
b_layer6_fc2=tf.Variable(tf.constant(0.05,shape=[2]))

layer6_fc2=tf.matmul(layer5_fc1,w_layer6_fc2)+b_layer6_fc2

layer6_fc2=tf.nn.dropout(layer6_fc2,keep_prob=0.7)


# softmax层
y_pred=tf.nn.softmax(layer6_fc2,name='y_pred')
y_pred_cls=tf.argmax(y_pred,dimension=1)

correct_pred=tf.equal(y_pred_cls,y_true_cls)
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# 设损失，交叉熵层
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=layer6_fc2,labels=y_true)
loss=tf.reduce_mean(cross_entropy)

# 设损失优化器
optimizer=tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

total_interations=0

saver=tf.train.Saver()

def train(interations):
    global total_interations

    for i in range(total_interations,total_interations+interations):
        x_batch,y_true_batch,x_train_names,batch_cls= data.train.next_batch(batch_size)
        x_valid_batch,y_valid_batch,x_valid_names,valid_batch_cls=data.valid.next_batch(batch_size)

        feed_dict_train={x:x_batch,y_true:y_true_batch}
        feed_dict_valid={x:x_valid_batch,y_true:y_valid_batch}

        sess.run(optimizer,feed_dict=feed_dict_train)

        if i % (int(data.train.num_examples/batch_size)*10) == 0:
            valid_loss=sess.run(loss,feed_dict=feed_dict_valid)
            epoch=int(i/int(data.train.num_examples/batch_size))

            acc=sess.run(accuracy,feed_dict=feed_dict_train)
            valid_acc=sess.run(accuracy,feed_dict_valid)

            msg = "Training Epoch {0}--- iterations: {1}--- Training Accuracy: {2:>6.1%}, Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
            print(msg.format(epoch+1,i,acc,valid_acc,valid_loss))

            saver.save(sess,'./dogs-cats-model/dog-cat.ckpt',global_step=i)

    total_interations += interations

train(8000)