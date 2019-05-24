import main_han
import tensorflow as tf

def detect_name(namelist):
    test_x=[]
    for name in namelist:
        name_vec=[]
        for word in name:
            name_vec.append(main_han.voacb.get(word))
        while len(name_vec) < main_han.max_name_len:
            name_vec.append(0)

        test_x.append(name_vec)

    test_out=main_han.out_flat

    saver=tf.train.Saver(tf.global_variables())

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())

    saver.restore(sess,'./model_han/name2sex-30')

    predictions=tf.argmax(test_out,1)
    res=sess.run(predictions,{main_han.X:test_x,main_han.keep_drop:1.0})

    i=0
    for name in namelist:
        print(name,'女' if res[i] == 0 else '男')
        i += 1

detect_name(["韩亚宁","唐宇迪", "褚小花", "刘德华", "韩冬梅"])