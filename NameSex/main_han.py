import csv
import tensorflow as tf
csvfile = 'name.csv'

trian_x=[]
train_y=[]
count1,count2=0,0
with open(csvfile,"r",encoding="utf-8") as dataset:
    read = csv.reader(dataset)

    for sample in read:
        if len(sample)==2 and sample[0] =='姓名' and sample[1] =='性别':
            continue

        if (len(sample) == 2):
            trian_x.append(sample[0])
            if (sample[1] == '男'):
                train_y.append([0, 1])
            else:
                train_y.append([1, 0])

        count1 += 1

max_name_len=max([len(name) for name in trian_x])
# print(max_name_len)
if max_name_len < 8:
    max_name_len=8

# 字频统计字典
vocabulary={}
for name in trian_x:
    count2 += 1
    tokens =[word for word in name]
    for word in tokens:
        if word in vocabulary:
            vocabulary[word] += 1
        else:
            vocabulary[word] = 1


vocabulary_list=[' '] + sorted(vocabulary,key=vocabulary.get,reverse=True)

vocabulary_list.pop(0)


# 按降序编码，就是编序号
voacb=dict([(x,y) for (y,x) in enumerate(vocabulary_list)])


train_x_vec=[]
for name in trian_x:
    name_vec=[]
    for word in name:
        name_vec.append(voacb.get(word))
    while len(name_vec) < max_name_len:
        name_vec.append(0)
    train_x_vec.append(name_vec)

print(len(train_x_vec))
print(len(trian_x))

# 构建网络模型
input_size=max_name_len
num_classes=2

batch_size=64
num_batch=len(train_x_vec) // batch_size

X=tf.placeholder(tf.int32,[None,input_size])
Y=tf.placeholder(tf.float32,[None,num_classes])

keep_drop=tf.placeholder(tf.float32)

vocabulary_size=len(vocabulary_list)
embedding_size=128
num_filters=128

with tf.name_scope("embedding"):
    W_embedding=tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
    embedding_chars=tf.nn.embedding_lookup(W_embedding,X)
    embedding_chars_expanded=tf.expand_dims(embedding_chars,-1)

filter_sizes=[2,3,4]
out_pools=[]

for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv_maxpool-%s"%filter_size):
        filter_shape=[filter_size,embedding_size,1,num_filters]

        W_conv=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1))
        b_conv=tf.Variable(tf.constant(0.1,shape=[num_filters]))
        # out_conv=tf.nn.relu(tf.nn.conv2d(embedding_chars_expanded,W_conv,strides=[1,1,1,1],padding="VALID")+b_conv)
        out_conv=tf.nn.conv2d(embedding_chars_expanded,W_conv,strides=[1,1,1,1],padding="VALID")
        out_relu=tf.nn.relu(tf.nn.bias_add(out_conv,b_conv))
        # conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")
        # h = tf.nn.relu(tf.nn.bias_add(conv, b))

        out_pool=tf.nn.max_pool(out_relu,ksize=[1,input_size-filter_size+1,1,1],strides=[1,1,1,1],padding="VALID")

        out_pools.append(out_pool)

num_filters_total=num_filters *len(filter_sizes)

h_pool=tf.concat(out_pools,3)
h_pool_flat=tf.reshape(h_pool,[-1,num_filters_total])

with tf.name_scope("dropout"):
    h_drop=tf.nn.dropout(h_pool_flat,keep_drop)

with tf.name_scope("output"):
    # W_flat=tf.get_variable("W",shape=[num_filters_total,num_classes],initializer=tf.contrib.layers.xavier_xavier_initializer())
    W_flat=tf.Variable(tf.truncated_normal([num_filters_total,num_classes],stddev=0.1))
    b_flat=tf.Variable(tf.constant(0.1,shape=[num_classes]))
    out_flat=tf.nn.xw_plus_b(h_drop,W_flat,b_flat)

with tf.name_scope("optimizer"):

    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_flat,labels=Y))

    optimizer = tf.train.AdamOptimizer(1e-3)

    grads=optimizer.compute_gradients(loss)
    train_op=optimizer.apply_gradients(grads)

saver=tf.train.Saver(tf.global_variables())

with tf.name_scope("train_op"):
    sess =tf.Session()
    sess.run(tf.global_variables_initializer())

    for e in range(100):
        for i in range(num_batch):
            batch_x=train_x_vec[i*batch_size : (i+1)*batch_size]
            batch_y=train_y[i*batch_size : (i+1)*batch_size]

            # _,loss_tmp = sess.run([train_op,loss], feed_dict={X:batch_x, Y:batch_y, keep_drop:0.5})
            _, loss_ = sess.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y, keep_drop: 0.5})
            if i % 1000 == 0:
                print("epoch:",e,"iter:",i,"loss",loss_)

        if e % 10 == 0:
            saver.save(sess,"./model_han/name2sex",global_step=e)
