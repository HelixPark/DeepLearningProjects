import os
import glob
import cv2
import numpy as np
from sklearn.utils import shuffle

class DataSet():
    def __init__(self,images,labels,images_names,cls):
        self._num_examples=images.shape[0]

        self._images=images
        self._labels=labels
        self._images_names=images_names
        self._cls=cls

        self._epochs_done=0
        self._index_in_epoch=0

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def images_names(self):
        return self._images_names

    @property
    def cls(self):
        return self._cls

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self,batch_size):
        start=self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch>self._num_examples:
            self._epochs_done +=1
            start = 0
            self._index_in_epoch=batch_size
            assert batch_size<=self._num_examples
        end = self._index_in_epoch

        return self._images[start:end],self._labels[start:end],self._images_names[start:end],self._cls[start:end]



def load_data(data_path,img_size,classes):
    # 存放图片、标签、名字，类别，相对应
    images=[]
    labels=[]
    images_names=[]
    cls=[]

    # 读文件夹,
    # fields有两种选择cats dogs,
    # index指在文件夹中序号dog为0，cat为1
    for fields in classes:
        index=classes.index(fields)
        # 设定路径
        path=os.path.join(data_path,fields,'*g')
        # 拿到此文件夹下所有的文件
        files=glob.glob(path)
        for fl in files:
            # 读图片，设置像素大小
            # 读出来的img为三维矩阵
            img=cv2.imread(fl)
            # 修改大小
            img=cv2.resize(img,(img_size,img_size),0,0,cv2.INTER_LINEAR)
            # 转换格式
            img=img.astype(np.float32)
            # 归一化操作
            img=np.multiply(img,1.0/255.0)
            images.append(img)

            # 读标签,独热表示，建立类别数大小的矩阵
            lbl=np.zeros(len(classes))
            # 根据索引设置labels的独热表示
            lbl[index]=1.0
            labels.append(lbl)

            # 读名字，用TensorBoard用得着
            # basename返回路径的最后文件名
            flbase=os.path.basename(fl)
            images_names.append(flbase)
            # 读类别
            cls.append(fields)
    # 读完之后，cats dogs都在里面
    # 把各类list转为ndarray格式
    images=np.array(images)
    labels=np.array(labels)
    images_names=np.array(images_names)
    cls=np.array(cls)

    return images,labels,images_names,cls

def read_data(train_data_path,img_size,classes,validation_rate):

    images,labels,images_names,cls=load_data(train_data_path,img_size,classes)
    # 打乱顺序，确保对应关系
    images,labels,images_names,cls=shuffle(images,labels,images_names,cls)

    if isinstance(validation_rate,float):
        validation_size=int(validation_rate*images.shape[0])
    # 构造验证集
    validation_images=images[:validation_size]
    validation_labels=labels[:validation_size]
    validation_images_names=images_names[:validation_size]
    validation_cls=cls[:validation_size]

    # 构造训练集
    train_images=images[validation_size:]
    train_labels=labels[validation_size:]
    train_images_names=images_names[validation_size:]
    train_cls=cls[validation_size:]

    class DataSets(object):
        pass
    data_sets=DataSets()

    # 定义DateSet类主要是里面的nextbatch方法
    data_sets.train=DataSet(train_images,train_labels,train_images_names,train_cls)
    data_sets.valid=DataSet(validation_images,validation_labels,validation_images_names,validation_cls)

    return data_sets
