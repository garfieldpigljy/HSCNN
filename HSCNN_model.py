import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HSCNN_network(nn.Module):
    def __init__(self,output_size, in_channels,out_channels,kernel_size,stride,padding,keep_probab,vocab_size,embedding_size,d):

        super(HSCNN_network, self).__init__()

        """
        :param batch_size: Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        :param output_size: number of classes
        :param in_channel: number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_size)
        :param out_channel: Number of output channels after convolution operation performed on the input matrix

        :param kernel_size: A list consisting of 3 different kernel_size. Convolution will be performed 3 times and finally results from each kernel_size will be concatenated.
        :param keep_probab: Probability of retaining an activation node during dropout operation
        :param vocab_size: Size of the vocabulary containing unique words
        :param embedding_size: Embedding dimension of GloVe word embeddings  dim =300
        :param weights:  Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
        """

        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.vocab_size1 = vocab_size
        self.embedding_size1 = embedding_size
        self.keep_probab = keep_probab
        self.d = d

        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_size[0], embedding_size), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_size[1], embedding_size), stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_size[2], embedding_size), stride, padding)
        self.dropout = nn.Dropout(keep_probab)
        self.liner = nn.Linear(len(kernel_size)*out_channels, 1024)
        self.label = nn.Linear(1024, 103)
        self.sig = nn.Sigmoid()
        self.out = nn.Linear(1024, output_size)
        self.liner_onehot = nn.Linear(103, 1024)
        self.rel = nn.ReLU()

    def conv_pool(self, input, conv_layer):
        conv_out = conv_layer(input)
        activation = F.relu(conv_out.squeeze(3))
        pool_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        return pool_out

    def forward_one(self, input_sentences, onehot):

        input_sentences = input_sentences.transpose(1, 0)
        input = self.word_embeddings(input_sentences)
        input = self.dropout1(input)
        input = input.unsqueeze(1)

        pool_out1 = self.conv_pool(input, self.conv1)
        pool_out2 = self.conv_pool(input, self.conv2)
        pool_out3 = self.conv_pool(input, self.conv3)
        all_pool_out = torch.cat((pool_out1, pool_out2, pool_out3), 1)
        x = self.dropout(all_pool_out)
        self.dropout1 = nn.Dropout(0.25)

        x = self.liner(x)
        cnn_out = self.label(x)

        q_w = self.liner_onehot(onehot)
        q_w = self.rel(q_w / math.sqrt(self.d))

        return x, q_w, cnn_out

    def forward_sia(self, input1, input2, onehot1, onehot2, state1):

        if state1 == 'train':
            x1, q_w1, cnn_out1 = self.forward_one(input1, onehot1)
            out1 = self.sig(x1)
            x2, q_w2, cnn_out2 = self.forward_one(input2, onehot2)
            out2 = self.sig(x2)

            dis = torch.abs(out1 - out2)
            tmp = torch.mul(dis, q_w2)
            out = self.out(tmp)

            return out1, out2, cnn_out1, cnn_out2, out

        if state1 == 'prediction':
            out1 = input1
            out2 = input2

            dis = torch.abs(out1 - out2)
            q_w2 = self.liner_onehot(onehot2)
            tmp = torch.mul(dis, q_w2)
            out = self.out(tmp)

            return out1, out2, out