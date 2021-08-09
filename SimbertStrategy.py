#! -*- coding: utf-8 -*-
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import DataGenerator
from keras.models import Model
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
import pandas as pd
import faiss
import gc, os, time
import joblib
import mkl
import yaml

mkl.get_max_threads()
TESTING = False

class data_generator(DataGenerator):
    def __init__(self, data, dict_path, maxlen=32, batch_size=32, buffer_size=None):
        super(data_generator, self).__init__(data, batch_size=32, buffer_size=None)
        self.maxlen = maxlen
        self.dict_path = dict_path

    def __iter__(self, random=False):
        tokenizer = Tokenizer(self.dict_path, do_lower_case=True)
        batch_token_ids, batch_segment_ids = [], []
        for is_end, text in self.sample(random):
            token_id, segment_id = tokenizer.encode(str(text), maxlen=self.maxlen)
            batch_token_ids.append(token_id)
            batch_segment_ids.append(segment_id)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids]
                batch_token_ids, batch_segment_ids = [], []

    def forpred(self, random=False):
        while True:
            for d in self.__iter__(random):
                yield d


class SimbertStrategy():
    '''利用simbert和faiss索引构建检索系统的全流程'''
    def __init__(self, configs_enlarge_path="./config.enlarge_strategy.yaml"):
        self.configs_enlarge = yaml.load(open(configs_enlarge_path))
        self.index_param = self.configs_enlarge['EnlargeStrategyConfig']['simbert']['index_param']
        self.high_thr = self.configs_enlarge['EnlargeStrategyConfig']['simbert']['high_thr']
        self.low_thr = self.configs_enlarge['EnlargeStrategyConfig']['simbert']['low_thr']
        self.topK = self.configs_enlarge['EnlargeStrategyConfig']['simbert']['topK']
        self.is_pca = self.configs_enlarge['EnlargeStrategyConfig']['simbert']['is_pca']
        self.vec_dim = self.configs_enlarge['EnlargeStrategyConfig']['simbert']['vec_dim']
        self.pca_dim = self.configs_enlarge['EnlargeStrategyConfig']['simbert']['pca_dim']
        self.maxlen = self.configs_enlarge['EnlargeStrategyConfig']['simbert']['maxlen']
        self.batch_size = self.configs_enlarge['EnlargeStrategyConfig']['simbert']['batch_size']
        self.pca_path = self.configs_enlarge['EnlargeStrategyConfig']['simbert']['pca_save_path']
        self.pretrain_model_path = self.configs_enlarge['EnlargeStrategyConfig']['simbert']['pretrain_model_path']
        self.config_path = os.path.join(self.pretrain_model_path, 'bert_config.json')
        self.checkpoint_path = os.path.join(self.pretrain_model_path, 'bert_model.ckpt')
        self.encoder = self.__init_encoder__()

    def __init_encoder__(self):
        bert = build_transformer_model(self.config_path, self.checkpoint_path, with_pool='linear', application='unilm',
                                       return_keras_model=False, )
        encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
        return encoder

    def __tofloat32__(self, vecs):
        return vecs.astype(np.float32)

    def __normvec__(self, vecs):
        return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5

    def encode_text(self, text_list, is_norm=True):
        """
        通过SimBERT将文本转化成向量
        :param text_list:
        :param norm:
        :return:
        """
        dict_path = os.path.join(self.pretrain_model_path, 'vocab.txt')
        data_gen = data_generator(data=text_list, dict_path=dict_path, batch_size=self.batch_size, maxlen=self.maxlen)
        vecs = self.encoder.predict_generator(data_gen.forpred(), steps=len(data_gen), verbose=1)
        if is_norm:
            vecs = self.__normvec__(vecs)
        return self.__tofloat32__(vecs)

    def get_faiss_index(self):
        """
        构建索引，可以是单个索引参数，也可以是多个索引参数
        :param dim:向量维度
        :return:
        """
        vec_dim = self.pca_dim if self.is_pca else self.vec_dim
        if 'HNSW' in self.index_param and ',' not in self.index_param:
            hnsw_num = int(self.index_param.split('HNSW')[-1])
            index = faiss.IndexHNSWFlat(vec_dim, hnsw_num, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.index_factory(vec_dim, self.index_param, faiss.METRIC_INNER_PRODUCT)
            index.verbose = True
            index.do_polysemous_training = False
        return index

    def build_faiss_index(self, unlabel_query_path, index_save_path):
        df_unlable = pd.read_csv(unlabel_query_path)
        if TESTING:
            df_unlable = df_unlable.head(10000)
        # 文本数据列，也就是 item
        target_col = df_unlable.columns[0]
        text_vecs = self.encode_text(list(df_unlable[target_col]))
        # 是否需要进行PCA降维操作
        if self.is_pca:
            pca_model = PCA(n_components=self.pca_dim)
            pca_model.fit(text_vecs)
            joblib.dump(pca_model, self.pca_path)
            text_vecs = pca_model.transform(text_vecs)
            text_vecs = self.__tofloat32__(text_vecs)
        else:
            text_vecs = self.__tofloat32__(text_vecs)
        # 构建索引
        index = self.get_faiss_index()
        # 索引训练
        index.train(text_vecs)
        # 索引添加
        index.add(text_vecs)
        assert index.ntotal == len(df_unlable), '构建完成的index样本数需要和unlable_df样本数一致，请检查'
        print('索引构建完毕...')
        # 索引存储
        faiss.write_index(index, index_save_path)
        del index
        gc.collect()

    def query_search_by_faiss(self, index_save_path, label_path, unlabel_query_path, enlarge_query_save_path):
        """
        根据带标签数据集查找相似文本
        :param index_save_path:
        :param label_path:
        :param pca_path:
        :return:
        """
        search_start = time.time()
        label_df = pd.read_csv(label_path) if isinstance(label_path, str) else label_path
        unlabel_df = pd.read_csv(unlabel_query_path) if isinstance(unlabel_query_path, str) else unlabel_query_path
        if TESTING:
            label_df = label_df.head(10000)
        # 这里用于控制必须是带标签数据集
        assert 'label' in label_df.columns, f'输入的csv文件的column中必须要包含label这个字段'
        # 查找的字段，这里是item
        target_col = label_df.columns[0]
        text_vecs = self.encode_text(list(label_df[target_col]))
        if self.is_pca:
            # 读取pca模型
            pca_model = self.load_pca_model(self.pca_path)
            text_vecs = pca_model.transform(text_vecs)
        text_vecs = self.__tofloat32__(text_vecs)

        # 读取索引
        index = faiss.read_index(index_save_path)
        cos_values, sim_index_list = index.search(text_vecs, self.topK)
        search_end = time.time()
        print(f'Faiss检索共耗时：{search_end - search_start} s')
        del text_vecs
        gc.collect()

        sim_text_list = []
        target_col = label_df.columns[0]
        for k, (index_list, cos_list) in tqdm(enumerate(zip(sim_index_list, cos_values))):
            cos_list = np.array(cos_list)
            index_list = np.array(index_list)
            # 将sim_value卡在上下阈值内的index拿出来
            thr_id = (cos_list <= self.high_thr) & (cos_list >= self.low_thr)
            tmp_simcos_list = list(cos_list[thr_id])
            tmp_simtext_index_list = list(index_list[thr_id])
            if tmp_simtext_index_list and tmp_simcos_list:
                tmp_sim_text_df = pd.DataFrame(label_df.iloc[k]).T
                tmp_sim_text_df['sim_text'] = [list(unlabel_df[target_col].iloc[tmp_simtext_index_list])]
                tmp_sim_text_df['sim_value'] = [tmp_simcos_list]
                sim_text_list.append(tmp_sim_text_df)
        # 判断是否有满足阈值的匹配数据
        if len(sim_text_list)>0:
            df_sim_text = pd.concat(sim_text_list, axis=0).reset_index(drop=True)
            # 加工成['sim_text', 'sim_value', 'label']  格式数据
            df_sim_text['pair'] = df_sim_text.apply(lambda x : [[sim, cos] for sim, cos in zip(x['sim_text'], x['sim_value'])],axis=1)
            df_sim_text = df_sim_text.explode('pair').reset_index(drop=True)
            df_sim_text[['sim_text', 'sim_value']] = pd.DataFrame(df_sim_text['pair'].to_list(), columns=['sim_text', 'sim_value'])
            df_sim_text['sim_value'] = df_sim_text['sim_value'].values.astype(np.float32)
            df_sim_text = df_sim_text.drop(columns='pair')
            # 只选择正样本
            #df_sim_text = df_sim_text[df_sim_text['label']==1]
            df_sim_text = df_sim_text.sort_values(by='label', ascending=False).drop_duplicates(subset=['sim_text'], keep='first')
            df_sim_text[['sim_text', 'sim_value', 'label']].to_csv(enlarge_query_save_path, index=False)
        else:
            print("没有符合阈值的匹配数据")

    def cal_sim(self, query1, query2):
        '''计算两条query相似度得分'''
        vecs1, vecs2 = self.encode_text([query1, query2])
        sim_score = np.dot(vecs1, vecs2)
        return sim_score

    def cal_list_sim(self, query1, query2_list):
        '''计算query和一个querylist内所有的相似度并排序'''
        vecs = self.encode_text([query1] + query2_list)
        vecs1, vecs_list = vecs[0], vecs[1:]
        sim_score_list = list(vecs1.dot(vecs_list.T))
        sim_df = pd.DataFrame([query2_list]).T
        sim_df.columns=['query']
        sim_df['score'] = sim_score_list
        sim_df = sim_df.sort_values(by="score", ascending=False)
        return sim_df

    def get_enlarge_df(self, index_save_path, label_query_path, unlabel_query_path, enlarge_data_path):
        """
        根据带标签数据集label_query_path和无标签数据集unlabel_query_path来enlarge数据集
        先根据unlabel_query_path构建索引
        然后找label_query_path相似文本存储到enlarge_data_path，数据格式为['sim_text', 'sim_value', 'label']
        """
        # 判断是否有索引
        if(not os.path.isfile(index_save_path)):
            # 构建索引
            print("开始构建索引")
            self.build_faiss_index(unlabel_query_path, index_save_path)

        # 查找索引
        print("根据索引查找相似数据")
        self.query_search_by_faiss(index_save_path, label_query_path, unlabel_query_path, enlarge_data_path)

if __name__ == "__main__":
    # enlarge的时候可以 扩展训练集  也可以扩展测试集获取更好的数据分布
    label_query_path = '/faiss_ann/lable_dataset.csv'                           #有标签数据集路径
    unlabel_query_path = '/faiss_ann/unlable_dataset.csv'
    index_save_path = '/faiss_ann/unlable_dataset.csv.index'
    enlarge_query_save_path = '/faiss_ann/enlarge.csv'

    # 这是我需要的enlarge接口
    simbert_strategy = SimbertStrategy()
    print(index_save_path)
    print(label_query_path)
    print(unlabel_query_path)
    print(enlarge_query_save_path)
    simbert_strategy.get_enlarge_df(index_save_path, label_query_path, unlabel_query_path, enlarge_query_save_path)
