import torch
from load_data import load_label_list

def get_compare(train_data, model):
    model.eval()
    train_result = []

    with torch.no_grad():
        for batch in train_data:
            text = batch.text
            label1 = batch.onehot

            if torch.cuda.is_available():
                text = text.cuda()
                label1 = label1.cuda()

            out1, out2, cnn_out1, cnn_out2, out = model.forward_sia(text, text,  label1, label1, state1='train')
            sum = 0
            for item in out1:
                sum += item
            sum_tmp_result = (sum)/5

            train_result.append(sum_tmp_result)

    return train_result

def predict(test_data, train_data, train_result, model):
    codes_list = load_label_list()
    model.eval()
    TP = []
    FP = []
    FN = []
    leaf_TP = []
    leaf_FP = []
    leaf_FN = []
    small_TP = []
    small_FP = []
    small_FN = []
    for i in range(len(codes_list)):
        TP.append(0)
        FP.append(0)
        FN.append(0)
        leaf_TP.append(0)
        leaf_FP.append(0)
        leaf_FN.append(0)
        small_TP.append(0)
        small_FP.append(0)
        small_FN.append(0)
    with torch.no_grad():
        train_list = []

        for batch_id, batch in enumerate(train_data):
            label1 = batch.onehot
            label1 = label1.cuda()
            train_list.append(label1[0])

        for batch_id, batch in enumerate(test_data):
            text = batch.text
            label = batch.label
            label2 = batch.onehot

            if torch.cuda.is_available():
                text = text.cuda()
                label = label.cuda()
                label2 = label2.cuda()

                out1, out2, cnn_out1, cnn_out2, out = model.forward_sia(text, text, label2, label2, state1='train')
                input2 = out1

                result = []
                for vec in input2:
                    tmp = []
                    for i in range(len(train_result)):
                        item = train_result[i]
                        label2 = train_list[i]
                        out1, out2, out = model.forward_sia(vec, item, label2, label2, state1='prediction')
                        sout = torch.sigmoid(out)
                        a = sout + 0.5
                        b = a.int()
                        tmp.append(b)
                    result.append(tmp)


                for i in range(len(result)):
                    for j in range(len(result[i])):
                        if result[i][j] == 1 and int(float(label[i][j])) == 1:
                            TP[j] += 1
                        if result[i][j] == 1 and int(float(label[i][j])) == 0:
                            FP[j] += 1
                        if result[i][j] == 0 and int(float(label[i][j])) == 1:
                            FN[j] += 1

                print('[{}/{}]\t'.format(batch_id, len(test_data) - 1))


        ### Micro F1
        if sum(TP) == 0 and sum(FP) == 0:
            precision = 0
        else:
            precision = sum(TP)/(sum(TP)+sum(FP))
        if sum(TP) == 0 and sum(FN) == 0:
            recall = 0
        else:
            recall = sum(TP)/(sum(TP)+sum(FN))
        if precision == 0 and recall == 0:
            Micro_F1 = 0
        else:
            Micro_F1 = 2*precision*recall/(precision+recall)

        tmp = []
        for i in range(103):
            if TP[i] == 0 and FP[i] == 0:
                precision1 = 0
            else:
                precision1 = TP[i]/(TP[i]+FP[i])
            if TP[i] == 0 and FN[i] == 0:
                recall1 = 0
            else:
                recall1 = TP[i]/(TP[i]+FN[i])
            if precision1 == 0 and recall1 == 0:
                result1 = 0
            else:
                result1 = 2*precision1*recall1/(precision1+recall1)
            tmp.append(result1)

        total = 0
        for item in tmp:
            total += item
        Macro_F1 = total/len(codes_list)

        print('Micro F1:', Micro_F1)
        print('Macro F1', Macro_F1)



