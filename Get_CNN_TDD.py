#由于前期处理掉一些单词，导致在引入到CNN时，会造成大量的0, 影响结果，所以将所有词计算

import pickle
import numpy as np
import os
import csv

# 求tf与df比率的乘积
def both_ratio(inter_word_ratio, document_ratio):
    if len(inter_word_ratio) != len(document_ratio):
        print("both_ratio函数中类别数目出现问题")
        return

    tf_df = []
    for i in range(len(inter_word_ratio)):
        class_ratio = []
        if len(inter_word_ratio[i]) != len(document_ratio[i]):
            print("both_ratio函数中第 " + str(i) + " 类别数目出现问题")
            continue
        else:
            class_tf_ratio = inter_word_ratio[i]
            class_df_ratio = document_ratio[i]

            for j in range(len(class_tf_ratio)):
                r = class_df_ratio[j] * class_tf_ratio[j]
                class_ratio.append(r)

        tf_df.append(class_ratio)

    return tf_df


def Get_ALL_word_tf_df(savePath, tf_df, conter_word_list, idtoword):
    if not os.path.exists(savePath + 'cnn_word_list.pickle'):
        class_num = len(tf_df)
        for i in range(len(idtoword)):
            word = idtoword[i]

            if word in conter_word_list:
                continue
            else:
                conter_word_list.append(word)
                for j in range(class_num):
                    tf_df[j].append(1.0)

        pickle.dump(tf_df, open(savePath + 'cnn_tf_df.pickle', 'wb'))
        pickle.dump(conter_word_list, open(savePath + 'cnn_word_list.pickle', 'wb'))
    else:
        tf_df_file = open(savePath + 'cnn_tf_df.pickle', 'rb')
        tf_df = pickle.load(tf_df_file)

        conter_word_list_file = open(savePath + 'cnn_word_list.pickle', 'rb')
        conter_word_list = pickle.load(conter_word_list_file)

    return tf_df, conter_word_list


# ---------------------------------------------------------  向量化  --------------------------------------------
def load_class_embedding(wordtoidx, class_name, W_emb):
    print("load class embedding")
    name_list = [k.lower().split(' ') for k in class_name]
    id_list = [[wordtoidx[i] for i in l] for l in name_list]
    # W_emb = np.squeeze(W_emb, 0)
    value_list = [[W_emb[i] for i in l] for l in id_list]  # opt.W_emb:词表中每个词对应的词向量
    value_mean = [np.mean(l, 0) for l in value_list]  # 压缩行，对各列求均值
    return np.asarray(value_mean)


def generate_embedding(savePath, word_list, classes_Name, word2id, embpath, saveFilename=''):
    print("start word embedding")
    if not os.path.exists(savePath + saveFilename + 'save_word_embedding.pickle'):
        with open(embpath, 'rb') as emb_file:
            W_emb = pickle.load(emb_file)
            # W_emb = np.squeeze(W_emb, 0)

            id_list = []
            for l in word_list:
                if l not in word2id:
                    l = 'UNK'

                id_list.append(word2id[l])
            # id_list.append(id)

            # id_list = [[word2id[i] for i in l] for l in word_list]
            word_embedding = [W_emb[l] for l in id_list]  # opt.W_emb:词表中每个词对应的词向量

            pickle.dump(word_embedding, open(savePath + saveFilename + 'save_word_embedding.pickle', 'wb'))
    else:
        word_file = open(savePath + saveFilename + 'save_word_embedding.pickle', 'rb')
        word_embedding = pickle.load(word_file)

    print("start class embedding")
    if not os.path.exists(savePath + saveFilename + 'save_class_embedding.pickle'):
        with open(embpath, 'rb') as emb_file:
            W_emb = pickle.load(emb_file)

            classes_embedding_mean = load_class_embedding(word2id, classes_Name, W_emb)
            pickle.dump(classes_embedding_mean, open(savePath + saveFilename + 'save_class_embedding.pickle', 'wb'))

    else:
        class_mean_file = open(savePath + saveFilename + 'save_class_embedding.pickle', 'rb')
        classes_embedding_mean = pickle.load(class_mean_file)

    return word_embedding, classes_embedding_mean


# ---------------------------------------------------------  计算距离  --------------------------------------------
# 欧式距离 范围：[0,1]， 距离越小，相似度越大。
def calEuclideanDistance(vec1, vec2):
    # vec1 = np.array(vec1)
    # vec2 = np.array(vec2)

    sum_value = 0
    for i in range(len(vec1)):
        sum_value = sum_value + (vec1[i] - vec2[i]) * (vec1[i] - vec2[i])

    dist = np.linalg.norm(vec1 - vec2)

    return dist

#余弦距离， 距离越大越相似，值为1
# 夹角余弦取值范围为[-1,1]。夹角余弦越大表示两个向量的夹角越小，夹角余弦越小表示两向量的夹角越大。当两个向量的方向重合时夹角余弦取最大值1，当两个向量的方向完全相反夹角余弦取最小值-1
def calCosDistance(vector1, vector2):
    # vector1 = np.array([1, 2, 3])
    # vector2 = np.array([4, 7, 5])

    dist = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))

    return dist


def cal_distance(savePath, word_embedding, classes_embedding_mean, saveFilename=''):
    print("calculate the distance between word and category")
    if not os.path.exists(savePath + saveFilename + 'distance_of_words_and_categories.pickle'):
        dis = []

        # 计算词与类别之间的距离
        for i in range(len(classes_embedding_mean)):
            dis_class = []
            for j in range(len(word_embedding)):
                # d = calEuclideanDistance(classes_embedding_mean[i], word_embedding[j])
                d1 = calCosDistance(classes_embedding_mean[i], word_embedding[j])

                # if d == 0:
                #     d1 = 10000
                # else:
                #     d1 = 1 / d

                dis_class.append(d1)

            dis.append(dis_class)

        pickle.dump(dis, open(savePath + saveFilename + 'distance_of_words_and_categories.pickle', 'wb'))
    else:
        distance_mean_file = open(savePath + saveFilename + 'distance_of_words_and_categories.pickle', 'rb')
        dis = pickle.load(distance_mean_file)

    return dis


# ---------------------------------------------------------  计算特征 --------------------------------------------
def Generage_Feature(savePath, categories_ratio_of_tf, distance, saveFilename=''):
    print("calculate Features")
    if not os.path.exists(savePath + saveFilename + 'Features.pickle'):
        feature = []
        if len(categories_ratio_of_tf) != len(distance):
            print("Generage_Feature 函数中 The number of category different! ---> " + str(
                len(categories_ratio_of_tf)) + ' VS ' + str(len(distance)))
            return ''

        for i in range(len(categories_ratio_of_tf)):
            word_num1 = len(categories_ratio_of_tf[i])
            word_num2 = len(distance[i])
            if word_num1 != word_num2:
                print("Generage_Feature 函数中 categore-" + str(i) + "------The number of word different! ")
                return ''
            else:
                fts = []
                for j in range(len(categories_ratio_of_tf[i])):
                    rt = (categories_ratio_of_tf[i][j] + 1) * distance[i][j]

                    fts.append(rt)

                feature.append(fts)

        pickle.dump(feature, open(savePath + saveFilename + 'Features.pickle', 'wb'))
    else:
        feature_file = open(savePath + saveFilename + 'Features.pickle', 'rb')
        feature = pickle.load(feature_file)

    return feature

def write_file(savePath, conter_word_list, tf_df, distance, TDD):
    # 写入csv
    # 打开文件，追加a
    with open(savePath + "cnn_Word_messages" '.csv', 'a', newline='') as out:
        # 设定写入模式
        csv_write = csv.writer(out, dialect='excel')
        # 写入具体内容
        content = []
        content.append("Words")

        class_num = len(tf_df)
        for i in range(class_num):
            dis = "TDD_" + str(i)
            content.append(dis)
        content.append(' ')
        for i in range(class_num):
            dis = "tf_df_" + str(i)
            content.append(dis)
        content.append(' ')
        for i in range(class_num):
            dis = "distance_" + str(i)
            content.append(dis)

        csv_write.writerow(content)

        for j in range(len(conter_word_list)):
            content = []
            content.append(conter_word_list[j])

            for i in range(class_num):
                content.append(TDD[i][j])
            content.append(' ')
            for i in range(class_num):
                content.append(tf_df[i][j])
            content.append(' ')
            for i in range(class_num):
                content.append(distance[i][j])

            csv_write.writerow(content)

if __name__ == '__main__':
    data_Names = ['yelp', 'ag_news', 'dbpedia','yelp_full', 'yahoo']
    classes_Names = [['bad', 'good'],
                     ['World', 'Sports', 'Business', 'Science'],
                     ['Company', 'Educational Institution', 'Artist', 'Athlete', 'Office Holder', 'Mean Of Transportation', 'Building', 'Natural Place', 'Village', 'Animal', 'Plant', 'Album', 'Film', 'Written Work'],
                     ['worst', 'bad', 'middle', 'good', 'best'],
                     ['Society Culture', 'Science Mathematics', 'Health', 'Education Reference', 'Computers Internet', 'Sports', 'Business Finance', 'Entertainment Music', 'Family Relationships', 'Politics Government']]


    for i in range(len(data_Names)):
        f = data_Names[i]
        print(f)

        if f == 'yahoo':
            data_path = '/home/dell/PycharmProjects/NLP/Idea-1/New_leam_dataset/' + f + '.p'
            embpath = "/home/dell/PycharmProjects/NLP/Idea-1/New_leam_dataset/"+ f + "_glove.p"
        else:
            data_path = '/home/dell/PycharmProjects/NLP/Idea-1/New_leam_dataset/' + f + '.pickle'
            embpath = "/home/dell/PycharmProjects/NLP/Idea-1/New_leam_dataset/"+ f + "_glove.pickle"

        filePath = '/home/dell/PycharmProjects/NLP/Idea-1/Results/'
        savePath = filePath + f +'/08/cnn/'
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        # -------------------------------------------------------------- 1. 加载数据  --------------------------------------
        data_file = open(data_path, 'rb')
        _, _, _, _, _, _, wordtoid, idtoword = pickle.load(data_file)

        document_ratio_file = open(filePath + f + '/08/document_ratio.pickle', 'rb')
        document_ratio = pickle.load(document_ratio_file)

        conter_word_list_file = open(filePath + f + '/08/save_word_tf_message.pickle', 'rb')
        conter_word_list, inter_word_ratio,_ = pickle.load(conter_word_list_file)

        tf_df = both_ratio(inter_word_ratio, document_ratio)


        # -------------------------------------------  TDD  -----------------------------------------------------------------
        # 为以后在CNN中使用，将所有的词都包含
        cnn_tf_df, cnn_conter_word_list = Get_ALL_word_tf_df(savePath, tf_df, conter_word_list, idtoword)

        # (4) 词向量
        print("词向量")
        classes_Name = classes_Names[i]
        cnn_word_embedding, cnn_classes_embedding_mean = generate_embedding(savePath, cnn_conter_word_list, classes_Name,
                                                                            wordtoid, embpath, saveFilename='cnn_')

        # (5) 计算余弦距离
        cnn_distance_mean = cal_distance(savePath, cnn_word_embedding, cnn_classes_embedding_mean, saveFilename='cnn_')

        # (6) 计算特征（ ((词频/总词频)* 类内文档率 * 距离) ）
        features = Generage_Feature(savePath, cnn_tf_df, cnn_distance_mean, saveFilename='cnn_')
        # features的行为关键词，列为类别

        # (7) 写入文档
        write_file(savePath, cnn_conter_word_list, cnn_tf_df, cnn_distance_mean, features)

