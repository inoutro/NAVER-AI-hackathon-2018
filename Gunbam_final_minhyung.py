# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import os

import numpy as np
import torch



from torch.autograd import Variable
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F

import nsml
from dataset import MovieReviewDataset, preprocess
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML

import sys


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, filename)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data, _ = preprocess(raw_data, config.strmaxlen)
        model.eval()
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        output_prediction = model(preprocessed_data)
        point = output_prediction.data.squeeze(dim=1).tolist()
        # DONOTCHANGE: They are reserved for nsml
        
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        # print(list(zip(np.zeros(len(point)), point)))
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def collate_fn(data: list):
    """
    PyTorch DataLoader에서 사용하는 collate_fn 입니다.
    기본 collate_fn가 리스트를 flatten하기 때문에 벡터 입력에 대해서 사용이 불가능해, 직접 작성합니다.

    :param data: 데이터 리스트
    :return:
    """
    review = []
    label = []
    for datum in data:
        review.append(datum[0])
        label.append(datum[1])
    # 각각 데이터, 레이블을 리턴
    return review, np.array(label)


class BasicConv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(BasicConv1d, self).__init__()
        self.conv1 = nn.Conv1d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        #self.relu = nn.ReLU(inplace=False)
        self.ELU = nn.ELU(inplace=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ELU(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(BasicConv2d, self).__init__()
        self.conv2 = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        #self.relu = nn.ReLU(inplace=False)
        self.ELU = nn.ELU(inplace=False)

    def forward(self, x):
        x = self.conv2(x)
        x = self.ELU(x)
        return x

# 실제 1단
class Mixed_2d(nn.Module):
    def __init__(self):
        super(Mixed_2d, self).__init__()

        self.embedding_dim = 128
        self.branch0 = nn.Sequential(
            BasicConv2d(1, 16, kernel_size=(1,self.embedding_dim), stride=(1,1),padding=(0,0))
            # nn.AvgPool2d((2, 1), stride=2, padding=0, count_include_pad=False)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1, 8, kernel_size=(1,self.embedding_dim), stride=(1,1), padding=(0,0)),
            BasicConv2d(8, 16, kernel_size=(3,1), stride=(1,1), padding=(1,0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1, 8, kernel_size=(1,self.embedding_dim),  stride=(1,1), padding=(0,0)),
            BasicConv2d(8, 8, kernel_size=(3,1), stride=(1,1), padding=(1,0)),
            BasicConv2d(8, 16, kernel_size=(3,1), stride=(1,1), padding=(1,0))
        )

        self.branch3 =nn.Sequential(
            # nn.AvgPool2d((1,self.embedding_dim), stride=1, padding=0, count_include_pad=False),
            BasicConv2d(1, 8, kernel_size=(1, self.embedding_dim), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(8, 16, kernel_size=(5,1), stride=(1,1), padding=(2,0))
        )
    #16 * 4
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


# shortcut 2단 반복
class Block_rep(nn.Module):
    def __init__(self, scale=1.0):
        super(Block_rep, self).__init__()
        # self.embedding_dim = 32
        self.scale = scale

        self.branch0 = BasicConv1d(64, 16, kernel_size=1, stride=1,padding=0) # 8

        self.branch1 = nn.Sequential(
            BasicConv1d(64, 16, kernel_size=1, stride=1, padding=0),
            BasicConv1d(16, 16, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv1d(64, 16, kernel_size=1, stride=1,padding=0),
            BasicConv1d(16, 16, kernel_size=3, stride=1, padding=1),
            BasicConv1d(16, 16, kernel_size=3, stride=1, padding=1)
        )

        self.conv1d = nn.Conv1d(48, 64, kernel_size=1, stride=1,padding=0)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv1d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

# shortcut 실제 2단 / 차원 축소 k/2
class Block_a(nn.Module):
    def __init__(self, scale=1.0):
        super(Block_a, self).__init__()
        # self.embedding_dim = 32
        self.scale = scale
        # 32
        self.branch0 = nn.Sequential(
            BasicConv1d(64 ,16, kernel_size=1, stride=1,padding=0),
            BasicConv1d(16, 32, kernel_size=3, stride=1, padding=1),
            BasicConv1d(32, 64, kernel_size=3, stride=2, padding=1)
        )
        # 2
        self.branch1 = nn.Sequential(
            BasicConv1d(64 , 64 , kernel_size=3, stride=2,padding=1)
            # BasicConv1d(16, 16, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.MaxPool1d(3,2,padding=1)

        # self.conv1d = nn.Conv1d(self.embedding_dim * 2 , self.embedding_dim * 4 , kernel_size=1, stride=1)
        # self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1,x2 ), 1)
        # out = self.conv1d(out)

        return out

# shortcut 2단 반복
class Block_repb(nn.Module):
    def __init__(self, scale=1.0):
        super(Block_repb, self).__init__()
        # self.embedding_dim = 32
        self.scale = scale

        self.branch0 = BasicConv1d(192 , 48, kernel_size=1, stride=1,padding=0)

        self.branch1 = nn.Sequential(
            BasicConv1d(192, 48, kernel_size=1, stride=1, padding=0),
            BasicConv1d(48, 48, kernel_size=7, stride=1, padding=3)
        )

        self.conv1d = nn.Conv1d(96 , 192, kernel_size=1, stride=1,padding=0)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        # x2 = self.branch2(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv1d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class BiRNN(nn.Module):
    def __init__(self, args, embedding_dim: int, max_length: int, hidden_size, num_layers):
        super(BiRNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_size = 251
        self.output_dim = 1  # Regression
        self.max_length = max_length
        # self.batch_size = batch_size

        self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)


    
        self.Mixed_2d = Mixed_2d()

        self.repeat_1 = nn.Sequential(
            Block_rep(scale=0.2),
            Block_rep(scale=0.2),
            Block_rep(scale=0.2),
            Block_rep(scale=0.2),
            Block_rep(scale=0.2)
        )
        self.Block_a = Block_a()

        self.repeat_2 = nn.Sequential(
            Block_repb(scale=0.2),
            Block_repb(scale=0.2),
            Block_repb(scale=0.2),
            Block_repb(scale=0.2),
            Block_repb(scale=0.2)
        )

        self.fc11 = nn.Sequential(
            nn.Linear(64, 52),
            nn.AlphaDropout(p=0.6),
            nn.ELU())
        # print(self.fc1)
        self.fc12 = nn.Sequential(
            nn.Linear(52, 48),
            nn.AlphaDropout(p=0.6),
            nn.ELU())
        ###
        self.fc1 = nn.Sequential(
            nn.Linear(192, 96),
            nn.AlphaDropout(p=0.6),
            nn.ELU())
        # print(self.fc1)

        self.fc2 = nn.Sequential(
            nn.Linear(96,48),
            nn.AlphaDropout(p=0.6),
            nn.ELU())

        self.fc3 = nn.Linear(48, 1)

    def forward(self, data: list):
        """

        :param data: 실제 입력값
        :return:
        """
        # 임베딩의 차원 변환을 위해 배치 사이즈를 구합니다.
        batch_size = len(data)
        # list로 받은 데이터를 torch Variable로 변환합니다.
        data_in_torch = Variable(torch.from_numpy(np.array(data)).long())
        # 만약 gpu를 사용중이라면, 데이터를 gpu 메모리로 보냅니다.
        if GPU_NUM:
            data_in_torch = data_in_torch.cuda()
        # 뉴럴네트워크를 지나 결과를 출력합니다.
        embeds = self.embeddings(data_in_torch)
        # print(embeds)

        # Set initial states

        if GPU_NUM:
            h0 = Variable(torch.zeros(self.num_layers*2, embeds.size(0), self.hidden_size)).cuda()
            c0 = Variable(torch.zeros(self.num_layers*2, embeds.size(0), self.hidden_size)).cuda()
        else:
            h0 = Variable(torch.zeros(self.num_layers*2, embeds.size(0), self.hidden_size)) # 2 for bidirection 
            c0 = Variable(torch.zeros(self.num_layers*2, embeds.size(0), self.hidden_size))


        if epoch < 5:
            # Forward propagate RNN
            embeds1, _ = self.lstm1(embeds, (h0, c0))
            embeds1 = embeds1.unsqueeze(1)
            mix = self.Mixed_2d(embeds1).squeeze(3)
            max_r2 = nn.functional.max_pool1d(mix, mix .size(2)).squeeze(2)
            output1 = self.fc11(max_r2)
            output2 = self.fc12(output1)

            output = torch.sigmoid(self.fc3(output2)/2 + 1) * 9 + 1


            return output

        if 8 > epoch >= 5:
            # Forward propagate RNN

            embeds1, _ = self.lstm1(embeds,(h0, c0))
            embeds1 = embeds1.unsqueeze(1)
            mix = self.Mixed_2d(embeds1).squeeze(3)

            r1 = self.repeat_1(mix)

            max_r2 = nn.functional.max_pool1d(r1, r1.size(2)).squeeze(2)
            output1 = self.fc11(max_r2)
            output2 = self.fc12(output1)

            # 영화 리뷰가 1~10점이기 때문에, 스케일을 맞춰줍니다
            output = torch.sigmoid(self.fc3(output2)/2 + 1) * 9 + 1
            return output


        if epoch >= 8:
            # Forward propagate RNN

            embeds1, _ = self.lstm1(embeds,(h0, c0))
            embeds1 = embeds1.unsqueeze(1)

            mix = self.Mixed_2d(embeds1).squeeze(3)

            r1 = self.repeat_1(mix)
            ba = self.Block_a(r1)
            r2 = self.repeat_2(ba)

            max_r2 = nn.functional.max_pool1d(r2, r2.size(2)).squeeze(2)
            output1 = self.fc1(max_r2)
            output2 = self.fc2(output1)

            # 영화 리뷰가 1~10점이기 때문에, 스케일을 맞춰줍니다
            output = torch.sigmoid(self.fc3(output2)/2 + 1) * 9 + 1
            return output
        


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch', type=int, default=3000)
    args.add_argument('--strmaxlen', type=int, default=200)
    args.add_argument('--embedding', type=int, default=8)
    args.add_argument('--initial_lr', type=float, default=0.001)

    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'


    model = BiRNN(args, config.embedding, config.strmaxlen, 64, 2)
    if GPU_NUM:
        model = model.cuda()


    # DONOTCHANGE: Reserved for nsml use
    bind_model(model, config)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.initial_lr)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())


    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=config.batch,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=2)
        total_batch = len(train_loader)
        # print(total_batch)
        # print(total)
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            for i, (data, labels) in enumerate(train_loader):
                predictions = model(data)
                label_vars = Variable(torch.from_numpy(labels))
                if GPU_NUM:
                    label_vars = label_vars.cuda()
                loss = criterion(predictions, label_vars)
                if GPU_NUM:
                    loss = loss.cuda()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Batch : ', i + 1, '/', total_batch,
                      ', MSE in this minibatch: ', loss.data[0])
                avg_loss += loss.data[0]
            print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch))
            # nsml ps, 혹은 웹 상의 텐서보드에 나타나는 값을 리포트하는 함수입니다.
            #
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss/total_batch), step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)