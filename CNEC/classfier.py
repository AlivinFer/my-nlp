# date: 03/24/2020
# coding: gbk
import numpy as np
from sklearn.model_selection import train_test_split
from CNEC.normalization import normalize_corpus
from CNEC.feature_extractors import bow_extractor, tfidf_extractor
import gensim
import jieba
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression


def get_data():
    """
    ��ȡ����
    :return:  �ı����ݣ���Ӧ��labels
    """
    with open("data/ham_data.txt", encoding='utf-8') as ham_f, open("data/spam_data.txt",
                                                                    encoding='utf-8') as spam_f:
        ham_data = ham_f.readlines()
        spam_data = spam_f.readlines()
        ham_label = np.ones(len(ham_data)).tolist()  # tolist��������������ת��Ϊ�б�����
        spam_label = np.zeros(len(spam_data)).tolist()
        corpus = ham_data + spam_data
        labels = ham_label + spam_label
    return corpus, labels


def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    """
    :param corpus: �ı�����
    :param labels: �ı���ǩ
    :param test_data_proportion:  ���Լ�����ռ��
    :return: ѵ�����ݣ� �������ݣ� ѵ��labels�� ����labels
    """
    x_train, x_test, y_train, y_test = train_test_split(corpus, labels, test_size=test_data_proportion,
                                                        random_state=42)  # �̶�random_state��ÿ�����ɵ�������ͬ����ģ����ͬ��
    return x_train, x_test, y_train, y_test


def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for docs, label in zip(corpus, labels):
        if docs.strip():
            filtered_corpus.append(docs)
            filtered_labels.append(label)
    return filtered_corpus, filtered_labels


def get_metrics(true_labels, predicted_labels):
    print('׼ȷ��:', np.round(
        metrics.accuracy_score(true_labels,
                               predicted_labels),
        2))
    print('����:', np.round(
        metrics.precision_score(true_labels,
                                predicted_labels,
                                average='weighted'),
        2))
    print('�ٻ���:', np.round(
        metrics.recall_score(true_labels,
                             predicted_labels,
                             average='weighted'),
        2))
    print('F1�÷�:', np.round(
        metrics.f1_score(true_labels,
                         predicted_labels,
                         average='weighted'),
        2))


def train_predict_evaluate_model(classifier,
                                 train_features, train_labels,
                                 test_features, test_labels):
    # build model
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features)
    # evaluate model prediction performance
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)
    return predictions


def main():
    corpus, labels = get_data()
    print("�ܵ���������", len(labels))
    corpus, labels = remove_empty_docs(corpus, labels)
    print('����֮һ:', corpus[10])
    print('������label:', labels[10])
    label_name_map = ['�����ʼ�', '�����ʼ�']  # 0 1
    print('ʵ������:', label_name_map[int(labels[10])], label_name_map[int(labels[5900])])

    train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus,
                                                                            labels,
                                                                            test_data_proportion=0.3)

    # �����ݽ��й�һ��
    norm_train_corpus = normalize_corpus(train_corpus)
    norm_test_corpus = normalize_corpus(test_corpus)

    ''.strip()

    # �ʴ�ģ������
    bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)
    bow_test_features = bow_vectorizer.transform(norm_test_corpus)

    # tfidf ����
    tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
    tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)

    # tokenize documents
    tokenized_train = [jieba.lcut(text)
                       for text in norm_train_corpus]
    print(tokenized_train[2:10])
    tokenized_test = [jieba.lcut(text)
                      for text in norm_test_corpus]
    # build word2vec ģ��
    # model = gensim.models.Word2Vec(tokenized_train,
    #                                size=500,
    #                                window=100,
    #                                min_count=30,
    #                                sample=1e-3)

    mnb = MultinomialNB()
    svm = SGDClassifier(loss='hinge', n_iter_no_change=100)
    lr = LogisticRegression()

    # ���ڴʴ�ģ�͵Ķ������ر�Ҷ˹
    print("���ڴʴ�ģ�������ı�Ҷ˹������")
    mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb,
                                                       train_features=bow_train_features,
                                                       train_labels=train_labels,
                                                       test_features=bow_test_features,
                                                       test_labels=test_labels)

    # ���ڴʴ�ģ���������߼��ع�
    print("���ڴʴ�ģ���������߼��ع�")
    lr_bow_predictions = train_predict_evaluate_model(classifier=lr,
                                                      train_features=bow_train_features,
                                                      train_labels=train_labels,
                                                      test_features=bow_test_features,
                                                      test_labels=test_labels)

    # ���ڴʴ�ģ�͵�֧������������
    print("���ڴʴ�ģ�͵�֧��������")
    svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
                                                       train_features=bow_train_features,
                                                       train_labels=train_labels,
                                                       test_features=bow_test_features,
                                                       test_labels=test_labels)

    # ����tfidf�Ķ���ʽ���ر�Ҷ˹ģ��
    print("����tfidf�ı�Ҷ˹ģ��")
    mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)

    # ����tfidf���߼��ع�ģ��
    print("����tfidf���߼��ع�ģ��")
    lr_tfidf_predictions = train_predict_evaluate_model(classifier=lr,
                                                        train_features=tfidf_train_features,
                                                        train_labels=train_labels,
                                                        test_features=tfidf_test_features,
                                                        test_labels=test_labels)

    # ����tfidf��֧��������ģ��
    print("����tfidf��֧��������ģ��")
    svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)


if __name__ == '__main__':
    main()
