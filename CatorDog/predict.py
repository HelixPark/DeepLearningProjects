import os,cv2
import glob
import numpy as np
import tensorflow as tf

predit_path='predict_data'

images=[]
image_size=64
num_channels=3
images_name=[]

path=os.path.join(predit_path,'*g')
# 拿到p_path下的所有文件
files=glob.glob(path)
for fl in files:
    image=cv2.imread(fl)
    image=cv2.resize(image,(image_size,image_size),0,0,cv2.INTER_LINEAR)
    image=image.astype(np.float32)
    image=np.multiply(image,1.0/255.0)
    images.append(image)

    flbase=os.path.basename(fl)
    images_name.append(flbase)
images=np.array(images)
x_batch=images.reshape(len(images),image_size,image_size,num_channels)

sess=tf.Session()
# 读取网络结构
saver=tf.train.import_meta_graph('./dogs-cats-model/dog-cat.ckpt-7820.meta')
# 读取网络权重即第一个.data文件，但是读的时候不需要数字后面的后缀
saver.restore(sess,'./dogs-cats-model/dog-cat.ckpt-7820')

graph=tf.get_default_graph()

y_pred=graph.get_tensor_by_name("y_pred:0")


x=graph.get_tensor_by_name("x:0")
y_true=graph.get_tensor_by_name("y_true:0")

y_test_images=np.zeros((len(images),2))

# 喂数据
result=sess.run(y_pred,feed_dict={x:x_batch,y_true:y_test_images})

result_label=['dog','cat']
for i in range(len(images)):
    # print(result[lb,:].argmax())
    print(result_label[result[i,:].argmax()])
# print(result_label[result.argmax()])