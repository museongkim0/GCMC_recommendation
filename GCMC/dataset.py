import os
import copy
import glob
import shutil
import pandas as pd
import numpy as np
import argparse

import torch
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from sklearn.model_selection import train_test_split


class MCDataset(InMemoryDataset):
    
    # 데이터 zip파일 다운로드 url
    url = '{url}'

    def __init__(self, cfg, root, transform=None, pre_transform=None):
        self.feature=cfg.feature
        super(MCDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])    

    @property
    def num_relations(self):
        return self.data.edge_type.max().item() + 1

    @property
    def num_nodes(self):
        return self.data.x.shape[0]

    @property
    def raw_file_names(self):
        return ['Books_rating.csv']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
       

    def process(self):
        # 평점데이터 path지정
        path = os.path.join(self.raw_dir, 'Books_rating.csv')

        train_df,test_df, nums = self.create_df(path)

        train_idx, train_gt = self.create_gt_idx(train_df, nums)
        test_idx, test_gt = self.create_gt_idx(test_df, nums)

        train_df['item_id'] = train_df['item_id'] + nums['user']

        # 특징벡터 path지정
        vae_fv = np.load(os.path.join(self.raw_dir, 'vae_feature_vector.npy'))
        nn_fv = np.load(os.path.join(self.raw_dir, 'nn_feature_vector.npy'))
        pca_fv = np.load(os.path.join(self.raw_dir, 'pca_feature_vector.npy'))
        #graph1_fv = np.load(os.path.join(self.raw_dir, 'idea1_graph_vector.npy'))
        graph2_fv = np.load(os.path.join(self.raw_dir, 'idea2_graph_vector.npy'))

        if self.feature == 'nn':
          x = torch.Tensor(nn_fv)
          print("feature: nn")
        elif self.feature == 'vae':
          x = torch.Tensor(vae_fv)
          print("feature: vae")
        elif self.feature == 'pca':
          x = torch.Tensor(pca_fv)
          print("feature: pca")
        elif self.feature == 'graph1':
          #x = torch.Tensor(graph1_fv)
          print("feature: graph1")
        elif self.feature == 'graph2':
          x = torch.Tensor(graph2_fv)
          print("feature: graph2")
        else:
          x = torch.arange(nums['node'], dtype=torch.long) 
          print("feature: None")
    

        # Prepare edges
        edge_user = torch.tensor(list(train_df['user_id'].values))
        edge_item = torch.tensor(list(train_df['item_id'].values))
        edge_index = torch.stack((torch.cat((edge_user, edge_item), 0),
                                  torch.cat((edge_item, edge_user), 0)), 0)
        edge_index = edge_index.to(torch.long)

        edge_type = torch.tensor(list(train_df['relation']))
        edge_type = torch.cat((edge_type, edge_type), 0)

        edge_norm = copy.deepcopy(edge_index[1])
        for idx in range(nums['node']):
            count = (train_df == idx).values.sum()
            edge_norm = torch.where(edge_norm==idx,
                                    torch.tensor(count),
                                    edge_norm)
        edge_norm = (1 / edge_norm.to(torch.float))

        # Prepare data
        data = Data(x=x, edge_index=edge_index)
        data.edge_type = edge_type
        data.edge_norm = edge_norm
        data.train_idx = train_idx
        data.test_idx = test_idx
        data.train_gt = train_gt
        data.test_gt = test_gt
        data.num_users = torch.tensor([nums['user']])
        data.num_items = torch.tensor([nums['item']])
        
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def create_df(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = ['user_id','item_id','relation']

        df['relation'] = df['relation'] - 1

        nums = {'user': df.max()['user_id'] + 1,
                'item': df.max()['item_id'] + 1,
                'node': df.max()['user_id'] + df.max()['item_id'] + 2,
                'edge': len(df)}
        
        train_df, test_df = train_test_split(df,shuffle=True,test_size=0.2,random_state=42)
        train_df.index = range(0,len(train_df))
        test_df.index = range(0,len(test_df))
        
        return train_df, test_df, nums

    def create_gt_idx(self, df, nums):
        df['idx'] = df['user_id'] * nums['item'] + df['item_id']
        idx = torch.tensor(df['idx'])
        gt = torch.tensor(df['relation'])
        return idx, gt    

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data.pt'))
        return data[0]

    def __repr__(self):
        return '{}{}()'.format(self.name.upper(), self.__class__.__name__)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MCDataset(feature=self.feature, root='./data/book')
    data = dataset[0]
    print(data)
    data = data.to(device)
