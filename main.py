# -*- coding: utf-8 -*-
"""
Guoyin Wang

LEAM
"""

import os, sys, pickle
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import csv
import scipy.io as sio
from math import floor

from model import *
from utils import get_minibatches_idx, restore_from_save, tensors_key_in_file, prepare_data_for_emb, load_class_embedding

class Options(object):
    def __init__(self):
        self.GPUID = 0
        self.dataset = 'yahoo'
        self.fix_emb = True
        self.restore = False
        self.W_emb = None
        self.W_class_emb = None
        self.maxlen = 305
        self.n_words = None
        self.embed_size = 300
        self.lr = 1e-3
        self.batch_size = 100
        self.max_epochs = 20
        self.dropout = 0.5
        self.part_data = False
        self.portion = 1.0 
        self.save_path = "./" + self.dataset + "/save/"
        self.log_path = "./" + self.dataset + "/log/"
        self.print_freq = 100
        self.valid_freq = 100

        self.optimizer = 'Adam'
        self.clip_grad = None
        self.class_penalty = 1.0
        self.ngram = 55
        self.H_dis = 300


    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

def emb_classifier(x, x_mask, y, dropout, opt, class_penalty, G_ours, seq_len):
    # comment notation
    #  b: batch size, s: sequence length, e: embedding dim, c : num of class

    # 对输入文档进行编码
    x_emb, W_norm = embedding(x, opt)  #  b * s * e
    x_emb=tf.cast(x_emb,tf.float32)
    W_norm=tf.cast(W_norm,tf.float32)
    y_pos = tf.argmax(y, -1)

    # 对标签进行编码
    y_emb, W_class = embedding_class(y_pos, opt, 'class_emb') # b * e, c * e
    y_emb=tf.cast(y_emb,tf.float32)
    W_class=tf.cast(W_class,tf.float32)
    W_class_tran = tf.transpose(W_class, [1,0]) # e * c

    x_emb = tf.expand_dims(x_emb, 3)  # b * s * e * 1
    H_enc, G_conv2 = att_emb_ngram_encoder_maxout_keywords(x_emb, x_mask, G_ours, seq_len, opt)
    H_enc = tf.squeeze(H_enc) # b*e

    # H_enc=tf.cast(H_enc,tf.float32)
    logits = discriminator_2layer(H_enc, opt, dropout, prefix='classify_', num_outputs=opt.num_class, is_reuse=False)  # b * c

    logits_G = discriminator_2layer(G_conv2, opt, dropout, prefix='classify_', num_outputs=opt.num_class, is_reuse=True)
    logits_class = discriminator_2layer(W_class, opt, dropout, prefix='classify_', num_outputs=opt.num_class,
                                        is_reuse=True)
    prob = tf.nn.softmax(logits)
    class_y = tf.constant(name='class_y', shape=[opt.num_class, opt.num_class], dtype=tf.float32, value=np.identity(opt.num_class),)
    correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)) + \
           class_penalty * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=class_y, logits=logits_class)) + \
           class_penalty * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_G))

    global_step = tf.Variable(0, trainable=False)
    train_op = layers.optimize_loss(
        loss,
        global_step=global_step,
        optimizer=opt.optimizer,
        learning_rate=opt.lr)

    return accuracy, loss, train_op, W_norm, global_step


def main():
    # Prepare training and testing data
    opt = Options()
    main_Path = '/home/dell/桌面/GG/TDD/keyword/Our_method/dataset/'
    # load data
    if opt.dataset == 'yahoo':
        loadpath = "/home/dell/PycharmProjects/NLP/Idea-1/New_leam_dataset/yahoo.p"
        embpath = "/home/dell/PycharmProjects/NLP/Idea-1/New_leam_dataset/yahoo_glove.p"
        load_G_path = '/home/dell/PycharmProjects/NLP/Idea-1/Results/yahoo/08/cnn/yahoo_G.pickle'
        opt.num_class = 10
        opt.class_name = ['Society Culture',
                          'Science Mathematics',
                          'Health',
                          'Education Reference',
                          'Computers Internet',
                          'Sports',
                          'Business Finance',
                          'Entertainment Music',
                          'Family Relationships',
                          'Politics Government']
    elif opt.dataset == 'agnews':
        loadpath = "/home/dell/PycharmProjects/NLP/Idea-1/New_leam_dataset/ag_news.pickle"
        embpath = "/home/dell/PycharmProjects/NLP/Idea-1/New_leam_dataset/ag_news_glove.pickle"
        load_G_path = '/home/dell/PycharmProjects/NLP/Idea-1/Results/ag_news/08/cnn/ag_news_G.pickle'
        opt.num_class = 4
        opt.class_name = ['World',
                          'Sports',
                          'Business',
                          'Science']
    elif opt.dataset == 'dbpedia':
        loadpath = "/home/dell/PycharmProjects/NLP/Idea-1/New_leam_dataset/dbpedia.pickle"
        embpath = "/home/dell/PycharmProjects/NLP/Idea-1/New_leam_dataset/dbpedia_glove.pickle"
        load_G_path = '/home/dell/PycharmProjects/NLP/Idea-1/Results/dbpedia/08/cnn/dbpedia_G.pickle'
        opt.num_class = 14
        opt.class_name = ['Company',
                          'Educational Institution',
                          'Artist',
                          'Athlete',
                          'Office Holder',
                          'Mean Of Transportation',
                          'Building',
                          'Natural Place',
                          'Village',
                          'Animal',
                          'Plant',
                          'Album',
                          'Film',
                          'Written Work',
                          ]
    elif opt.dataset == 'yelp_full':
        loadpath = "/home/dell/PycharmProjects/NLP/Idea-1/New_leam_dataset/yelp_full.pickle"
        embpath = "/home/dell/PycharmProjects/NLP/Idea-1/New_leam_dataset/yelp_full_glove.pickle"
        load_G_path = '/home/dell/PycharmProjects/NLP/Idea-1/Results/yelp_full/08/cnn/yelp_full_G.pickle'
        opt.num_class = 5
        opt.class_name = ['worst',
                          'bad',
                          'middle',
                          'good',
                          'best']
    elif opt.dataset == 'yelp':
        loadpath = "/home/dell/PycharmProjects/NLP/Idea-1/New_leam_dataset/yelp.pickle"
        embpath = "/home/dell/PycharmProjects/NLP/Idea-1/New_leam_dataset/yelp_glove.pickle"
        load_G_path = '/home/dell/PycharmProjects/NLP/Idea-1/Results/yelp/08/cnn/yelp_G.pickle'
        opt.num_class = 2
        opt.class_name = ['bad',
                          'good']
    x = pickle.load(open(loadpath, "rb"))
    train, val, test = x[0], x[1], x[2] #将单词由数字表示，已做了分词工作，且句子长度尚未统一
    train_lab, val_lab, test_lab = x[3], x[4], x[5]#label 采用one-hot编码形式表示
    wordtoix, ixtoword = x[6], x[7]

    #加载权重G
    G_train, G_val, G_test = pickle.load(open(load_G_path, "rb"))

    del x
    print("load data finished")

    train_lab = np.array(train_lab, dtype='float32')
    val_lab = np.array(val_lab, dtype='float32')
    test_lab = np.array(test_lab, dtype='float32')    
    opt.n_words = len(ixtoword)
    if opt.part_data:
        #np.random.seed(123)
        train_ind = np.random.choice(len(train), int(len(train)*opt.portion), replace=False)
        train = [train[t] for t in train_ind]
        train_lab = [train_lab[t] for t in train_ind]
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.GPUID)

    print(dict(opt))
    print('Total words: %d' % opt.n_words)

    try:
        opt.W_emb = np.array(pickle.load(open(embpath, 'rb')),dtype='float32')
        opt.W_class_emb = load_class_embedding( wordtoix, opt)
    except IOError:
        print('No embedding file found.')
        opt.fix_emb = False

    with tf.device('/gpu:1'):
        x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.maxlen],name='x_')
        x_mask_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.maxlen],name='x_mask_')
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        y_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.num_class],name='y_')
        class_penalty_ = tf.placeholder(tf.float32, shape=())
        G_our = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.maxlen, opt.num_class], name='G_our')
        seq_len = tf.placeholder(tf.int32, shape=[opt.batch_size], name='sque_sentence_num')
        accuracy_, loss_, train_op, W_norm_, global_step = emb_classifier(x_, x_mask_, y_, keep_prob, opt, class_penalty_, G_our, seq_len)
    uidx = 0
    max_val_accuracy = 0.
    max_test_accuracy = 0.
    val_acc = 0.

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, )
    config.gpu_options.allow_growth = True
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        if opt.restore:
            try:
                t_vars = tf.trainable_variables()
                save_keys = tensors_key_in_file(opt.save_path)
                ss = set([var.name for var in t_vars]) & set([s + ":0" for s in save_keys.keys()])
                cc = {var.name: var for var in t_vars}
                # only restore variables with correct shape
                ss_right_shape = set([s for s in ss if cc[s].get_shape() == save_keys[s[:-2]]])

                loader = tf.train.Saver(var_list=[var for var in t_vars if var.name in ss_right_shape])
                loader.restore(sess, opt.save_path)

                print("Loading variables from '%s'." % opt.save_path)
                print("Loaded variables:" + str(ss))

            except:
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())

        try:
            for epoch in range(opt.max_epochs):
                print("Starting epoch %d" % epoch)
                kf = get_minibatches_idx(len(train), opt.batch_size, shuffle=True)
                for _, train_index in kf:
                    uidx += 1
                    sents = [train[t] for t in train_index]
                    G1 = [G_train[t] for t in train_index]
                    x_labels = [train_lab[t] for t in train_index]
                    x_labels = np.array(x_labels)
                    x_labels = x_labels.reshape((len(x_labels), opt.num_class))
                    x_batch, x_batch_mask, G_batch, seq_len_batch = prepare_data_for_emb(sents, G1, opt)

                    _, loss, step,  = sess.run([train_op, loss_, global_step], feed_dict={x_: x_batch, x_mask_: x_batch_mask, y_: x_labels, keep_prob: opt.dropout, class_penalty_:opt.class_penalty, G_our:G_batch, seq_len:seq_len_batch})

                    if uidx % opt.valid_freq == 0:
                        train_correct = 0.0
                        # sample evaluate accuaccy on 500 sample data
                        kf_train = get_minibatches_idx(500, opt.batch_size, shuffle=True)
                        for _, train_index in kf_train:
                            train_sents = [train[t] for t in train_index]
                            train_G = [G_train[t] for t in train_index]
                            train_labels = [train_lab[t] for t in train_index]
                            train_labels = np.array(train_labels)
                            train_labels = train_labels.reshape((len(train_labels), opt.num_class))
                            x_train_batch, x_train_batch_mask, G_train_batch, x_train_seq_len = prepare_data_for_emb(train_sents, train_G, opt)
                            train_accuracy = sess.run(accuracy_, feed_dict={x_: x_train_batch, x_mask_: x_train_batch_mask, y_: train_labels, keep_prob: 1.0, class_penalty_:0.0, G_our:G_train_batch, seq_len:x_train_seq_len})

                            train_correct += train_accuracy * len(train_index)

                        train_accuracy = train_correct / 500

                        print("Iteration %d: Training loss %f " % (uidx, loss))
                        print("Train accuracy %f " % train_accuracy)

                        if not os.path.exists(opt.dataset + '_Train_message.csv'):
                            with open(opt.dataset + '_Train_message.csv', 'a', newline='') as out:
                                # 设定写入模式
                                csv_write = csv.writer(out, dialect='excel')
                                # 写入具体内容
                                csv_write.writerow(["epoch", "Training loss", "Train accuracy"])
                                csv_write.writerow([epoch, loss, train_accuracy])
                        else:
                            with open(opt.dataset + '_Train_message.csv', 'a', newline='') as out:
                                # 设定写入模式
                                csv_write = csv.writer(out, dialect='excel')
                                csv_write.writerow([epoch, loss, train_accuracy])


                        val_correct = 0.0
                        kf_val = get_minibatches_idx(len(val), opt.batch_size, shuffle=True)
                        for _, val_index in kf_val:
                            val_sents = [val[t] for t in val_index]
                            val_Gs = [G_val[t] for t in val_index]
                            val_labels = [val_lab[t] for t in val_index]
                            val_labels = np.array(val_labels)
                            val_labels = val_labels.reshape((len(val_labels), opt.num_class))
                            x_val_batch, x_val_batch_mask, G_val_batch, x_val_seq_len = prepare_data_for_emb(val_sents, val_Gs, opt)
                            val_accuracy = sess.run(accuracy_, feed_dict={x_: x_val_batch, x_mask_: x_val_batch_mask, y_: val_labels, keep_prob: 1.0, class_penalty_:0.0, G_our:G_val_batch, seq_len:x_val_seq_len})

                            val_correct += val_accuracy * len(val_index)

                        val_accuracy = val_correct / len(val)
                        print("Validation accuracy %f " % val_accuracy)

                        #测试网络
                        test_correct = 0.0
                        kf_test = get_minibatches_idx(len(test), opt.batch_size, shuffle=True)
                        for _, test_index in kf_test:
                            test_sents = [test[t] for t in test_index]
                            test_Gs = [G_test[t] for t in test_index]
                            test_labels = [test_lab[t] for t in test_index]
                            test_labels = np.array(test_labels)
                            test_labels = test_labels.reshape((len(test_labels), opt.num_class))
                            x_test_batch, x_test_batch_mask, G_test_batch, x_test_seq_len = prepare_data_for_emb(test_sents, test_Gs,
                                                                                                 opt)

                            test_accuracy = sess.run(accuracy_, feed_dict={x_: x_test_batch, x_mask_: x_test_batch_mask, y_: test_labels, keep_prob: 1.0, class_penalty_: 0.0, G_our: G_test_batch, seq_len:x_test_seq_len})

                            test_correct += test_accuracy * len(test_index)
                        test_accuracy = test_correct / len(test)
                        print("Test accuracy %f " % test_accuracy)
                        # max_test_accuracy = test_accuracy
                        if test_accuracy > max_test_accuracy:
                            max_test_accuracy = test_accuracy
                            val_acc = val_accuracy
#                        max_test_accuracy = max(test_accuracy, max_test_accuracy)
#                        val_acc = val_accuracy

                        if val_accuracy > max_val_accuracy:
                            max_val_accuracy = val_accuracy
                            test_acc = test_accuracy

                        if not os.path.exists(opt.dataset + '_Classification_Results.csv'):
                            with open(opt.dataset + '_Classification_Results.csv', 'a', newline='') as out:
                                # 设定写入模式
                                csv_write = csv.writer(out, dialect='excel')
                                # 写入具体内容
                                csv_write.writerow(["epoch", "val_accuracy", "test_accuracy"])
                                csv_write.writerow([epoch, val_accuracy, test_accuracy])
                        else:
                            with open(opt.dataset + '_Classification_Results.csv', 'a', newline='') as out:
                                # 设定写入模式
                                csv_write = csv.writer(out, dialect='excel')
                                csv_write.writerow([epoch, val_accuracy, test_accuracy])

                print("Epoch %d: Max Test accuracy %f" % (epoch, max_test_accuracy))
                saver.save(sess, opt.save_path, global_step=epoch)
                
            print("Max Test accuracy %f , val accuracy %f " % (max_test_accuracy, val_acc))
            print("Max val accuracy %f , test accuracy %f" % (max_val_accuracy, test_acc))
            with open(opt.dataset + '_Classification_Results.csv', 'a', newline='') as out:
                # 设定写入模式
                csv_write = csv.writer(out, dialect='excel')
                csv_write.writerow(['Max Test accuracy:', max_test_accuracy, 'val accuracy', val_acc])
                csv_write.writerow(['Max val accuracy:', max_val_accuracy, 'test accuracy', test_acc])
        except KeyboardInterrupt:
            print('Training interupted')
            print("Max Test accuracy %f " % max_test_accuracy)
            with open(opt.dataset + '_Classification_Results.csv', 'a', newline='') as out:
                # 设定写入模式
                csv_write = csv.writer(out, dialect='excel')
                csv_write.writerow(['Max Test accuracy:', max_test_accuracy, 'val accuracy', val_acc])
                csv_write.writerow(['Max val accuracy:', max_val_accuracy, 'test accuracy', test_acc])

if __name__ == '__main__':
    main()
