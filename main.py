import torch
import load_data
import random
import numpy as np
from HSCNN_model import HSCNN_network
from HSCNN_train import train
from HSCNN_classifier import predict,get_compare

random.seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)
torch.backends.cudnn.deterministic = True


print('load data............')
train_pairs_iter, valid_pairs_iter, train_data_iter, test_data_iter = load_data.load_pairs()

embedding_size = 300
keep_probab = 0.5
kernel_size = [3, 4, 5]
vocab_size = 41993
patience = 5
d = 103

sia_model = HSCNN_network(output_size=1, in_channels=1, out_channels=128, kernel_size=kernel_size, stride=1, padding=0,
                    keep_probab=keep_probab, vocab_size=vocab_size, embedding_size=embedding_size,d = d)

if torch.cuda.is_available():
    sia_model.cuda()


siamese_model_dict = sia_model.state_dict()
cnn_model_dict = torch.load('./data/CNN_checkpoint.pt')
state_dict = {k: v for k, v in cnn_model_dict.items() if k in siamese_model_dict.keys()}
siamese_model_dict.update(state_dict)
sia_model.load_state_dict(siamese_model_dict)

print('Start training siamese networks............')
model, train_loss_record, val_loss_record = train(train_pairs_iter, valid_pairs_iter, sia_model, patience)

print('Start get_compare')
train_result = get_compare(train_data_iter, model)

print(len(train_result))
print('Start predict')
predict(test_data_iter, train_data_iter, train_result, model)


