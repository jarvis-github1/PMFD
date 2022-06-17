import torch
import ImageFeature
import TextFeature
import FinalClassifier
import FuseAllFeature
from LoadData import *
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import time
import os
# from visdom import Visdom
# import sklearn.metrics as metrics
# import seaborn as sns

# loss = Visdom()
# acc=Visdom()
# loss.line([[0.,0.,0.,0.,0.,0.,0.,0.]], [1], win='train_loss', opts=dict(title='loss', legend=['loss1','loss2','loss3','loss4','loss5','loss6','loss7','loss8']))
# acc.line([[0.,0.,0.,0.,0.,0.,0.,0.,0.]], [1], win='train_acc', opts=dict(title='acc', legend=['acc1','acc2','acc3','acc4','acc5','acc6','acc7','acc8','acc9']))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def create_folder(foldernames):
    current_position = "./model_sava/"
    foldername=str(current_position)+str(foldernames)+"/"
    # textname=str(current_position)+str(foldernames)+"/"+str(foldernames)+'.txt'
    isCreate=os.path.exists(foldername)
    if not isCreate:
        os.makedirs(foldername)

        print(str(foldername)+'is created')
    else:
        print('Already exist')
        return False


class Multimodel_image_text_g(torch.nn.Module):
    def __init__(self, lstm_dropout_rate, fc_dropout_rate):
        super(Multimodel_image_text_g, self).__init__()
        self.image = ImageFeature.ExtractImageFeature()
        self.text = TextFeature.ExtractTextFeature(TEXT_LENGTH, TEXT_HIDDEN, lstm_dropout_rate)
        self.fuse = FuseAllFeature.ModalityFusion_2(1024, 512)
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate, 512)

    def forward(self, image_feature, text_index):
        image_result, image_seq = self.image(image_feature)
        text_result, text_seq = self.text(text_index, image_result)
        fusion = self.fuse(image_result, image_seq, text_result, text_seq.permute(1, 0, 2))
        output = self.final_classifier(fusion)
        return output
def train(model5, train_loader, val_loader, loss_fn, optimizer5, number_of_epoch,i,file):
    F1_old = 0
    TP = TN = FN = FP = 0
    F1_5 = 0
    TP5 = TN5 = FN5 = FP5 = 0
    p5 = r5 = 0
    for epoch in range(number_of_epoch):

        model5_train_loss = 0
        cla_train_loss = 0
        cla_correct_train = 0
        # for i in range(1,9):
        #     model_train_loss[i]=0
        #     model_correct_train[i] = 0
        #     eval('model'+str(i)+'.train()')
        model5_correct_train = 0
        model5.train()


        right_num = data_num = count = 0
        dict={}
        for text_index, image_feature, group, id in train_loader:
            count += 1
            dataset_len = len(train_loader)
            data_num += train_loader.batch_size
            group = group.view(-1, 1).to(torch.float32).to(device)
            model5_pred = model5(image_feature.to(device),text_index.to(device))


            model5_loss = loss_fn(model5_pred, group)
            model5_train_loss += model5_loss

            model5_correct_train += (model5_pred.round() == group).sum().item()

            optimizer5.zero_grad()

            model5_loss.backward()

            optimizer5.step()

            def f1_score(TP, TN, FN, FP, model):
                TP += ((model.round() == 1) & (group == 1)).cpu().sum().item()
                TN += ((model.round() == 0) & (group == 0)).cpu().sum().item()
                FN += ((model.round() == 0) & (group == 1)).cpu().sum().item()
                FP += ((model.round() == 1) & (group == 0)).cpu().sum().item()
                try:
                    p = TP / (TP + FP)
                    r = TP / (TP + FN)
                    F1 = 2 * r * p / (r + p)
                except:
                    pass
                return TP, TN, FN, FP, p, r, F1

            try:
                TP5, TN5, FN5, FP5, p5, r5, F1_5 = f1_score(TP5, TN5, FN5, FP5, model5_pred)
            except:
                pass
                # try:
            #     # compute F1 precision准确率 recall召回率
            #     p = TP / (TP + FP)
            #     # p = torch.floor_divide(TP,TP+FP)
            #     r = TP / (TP + FN)
            #     # r = torch.floor_divide(TP, TP + FN)
            #     F1 = 2 * r * p / (r + p)
            # except:
            #     pass
            try:
                print('Training...epoch:%d/%d' % (epoch + 1, number_of_epoch))
                print('batch:%d/%d' % (count, dataset_len))
                print(
                    "pic_txt_guid:train_loss=%.5f train_acc=%.3f precision=%.5f recall=%.5f F1_score=%.5f" % (
                        model5_train_loss / data_num, model5_correct_train / data_num, p5, r5, F1_5))
                print('')
            except:
                print('Training...epoch:%d/%d' % (epoch + 1, number_of_epoch))
                print('batch:%d/%d' % (count, dataset_len))

                print("pic_txt_guid:train_loss=%.5f train_acc=%.3f " % (
                    model5_train_loss / data_num, model5_correct_train / data_num))
                print('')

        file.write(
            'pic_txt_guid:train_loss={} train_acc={} precision={} recall={} F1_score={} '.format(
                model5_train_loss / data_num,
                model5_correct_train / data_num,
                p5, r5, F1_5))
        file.write('\n')
        file.close()

        # #chart
        # 			loss.line([[model1_train_loss.item()/data_num,model2_train_loss.item()/data_num
        # 				,model3_train_loss.item()/data_num,model4_train_loss.item()/data_num,model5_train_loss.item()/data_num
        # 				,model6_train_loss.item()/data_num,model7_train_loss.item()/data_num,model8_train_loss.item()/data_num]],
        # 				[count], win='train_loss', update='append')
        # 			acc.line([[model1_correct_train/data_num,model2_correct_train/data_num
        # 				,model3_correct_train/data_num,model4_correct_train/data_num,model5_correct_train/data_num
        # 				,model6_correct_train/data_num,model7_correct_train/data_num,model8_correct_train/data_num,right_num/data_num]],
        # 				[count], win='train_acc', update='append')

        # learning_rate adjustment
        F1_new = test(model5, val_loader,file)
        
        # test(model1, model2, model3,  val_loader)
        if F1_new < F1_old:
            print('Learning rate changed')

            optimizer5.param_groups[0]['lr'] *= 0.8


        F1_old = F1_new

        print('learning_rate:', optimizer3.param_groups[0]['lr'])
        print('F1:', F1_new)
        print('')

        # sava model

        state5 = {'model': model5.state_dict(), 'optimizer': optimizer5.state_dict()}

        name='model'+str(i)+"/"

        torch.save(state5, './model_sava/' + name + 'model5.pth')




def test(model5,val_loader,file):

    valid_loss5 = 0


    correct_valid5 = 0


    model5.eval()
    right_num = data_num = count = 0


    F1 = 0

    TP5 = TN5 = FN5 = FP5 = 0
    F1_5 = 0
    p5 = r5= 0
    with torch.no_grad():
        for val_text_index, val_image_feature, val_grad, val_id in val_loader:
            count += 1
            dataset_len = len(val_loader)
            val_group = val_grad.view(-1, 1).to(torch.float32).to(device)

            model5_val_pred = model5(val_image_feature.to(device),val_text_index.to(device))
            # right_num += (result.round() == val_group).sum().item()
            data_num += val_loader.batch_size

            val_loss5 = loss_fn(model5_val_pred, val_group)


            valid_loss5 += val_loss5


            correct_valid5 += (model5_val_pred.round() == val_group).sum().item()


            def f1_score(TP, TN, FN, FP, model):
                p = r = F1 = 0
                TP += ((model.round() == 1) & (val_group == 1)).cpu().sum().item()
                TN += ((model.round() == 0) & (val_group == 0)).cpu().sum().item()
                FN += ((model.round() == 0) & (val_group == 1)).cpu().sum().item()
                FP += ((model.round() == 1) & (val_group == 0)).cpu().sum().item()
                try:
                    p = TP / (TP + FP)
                    r = TP / (TP + FN)
                    F1 = 2 * r * p / (r + p)
                    return TP, TN, FN, FP, p, r, F1
                except:
                    return TP, TN, FN, FP, p, r, F1
                # return TP, TN, FN, FP, p, r, F1

            # try:

            TP5, TN5, FN5, FP5, p5, r5, F1_5 = f1_score(TP5, TN5, FN5, FP5, model5_val_pred)

            # except:
            #     pass
            try:
                print('Validing...')
                print('batch:%d/%d' % (count, dataset_len))

                print(
                    "pic_txt_guid:train_loss=%.5f train_acc=%.3f precision=%.5f recall=%.5f F1_score=%.5f" % (
                        valid_loss5 / data_num, correct_valid5 / data_num, p5, r5, F1_5))
                print('')
            except:
                print('Validing...')
                print('batch:%d/%d' % (count, dataset_len))

                print("pic_txt_guid:train_loss=%.5f train_acc=%.3f " % (
                    valid_loss5 / data_num, correct_valid5 / data_num))
                print('')

    file.write(
            'v_pic_txt_guid:train_loss={} train_acc={} precision={} recall={} F1_score={} '.format(
                valid_loss5 / data_num,
                correct_valid5 / data_num,
                p5, r5, F1_5))
    file.write('\n')
    return F1


def model_load(name):

    checkpoint5 = torch.load('./model_sava/' + name + 'model5.pth')
    model5.load_state_dict(checkpoint5['model'])
    optimizer5.load_state_dict(checkpoint5['optimizer'])


    print('Model_Load completed\n')
    return model5, optimizer5


learning_rate_list = [0.001]
fc_dropout_rate_list = [0, 0.3, 0.9, 0.99]
lstm_dropout_rate_list = [0, 0.2, 0.4]
weight_decay_list = [0, 1e-6, 1e-5, 1e-4]

all_Data = my_data_set(data_set)
train_fraction = 0.8
val_fraction = 0.2
train_set, val_set, test_set = train_val_test_split(all_Data, train_fraction, val_fraction)
batch_size = 24
data_shuffle = True

# load data

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=data_shuffle)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=data_shuffle)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=data_shuffle)
play_loader = DataLoader(test_set, batch_size=1, shuffle=data_shuffle)

# start train
import itertools

if __name__ == "__main__":
    comb = itertools.product(learning_rate_list, fc_dropout_rate_list, lstm_dropout_rate_list, weight_decay_list)
    i =1
    for learning_rate, fc_dropout_rate, lstm_dropout_rate, weight_decay in list(comb)[10:]:
        print(
            f"learning rate={learning_rate} | fc dropout={fc_dropout_rate} | lstm dropout={lstm_dropout_rate} | weight decay={weight_decay}")
        # loss function
        loss_fn = torch.nn.BCELoss()
        # initilize the model

        model5 = Multimodel_image_text_g(lstm_dropout_rate, fc_dropout_rate).to(device)
        # optimizer
        optimizer5 = torch.optim.Adam(model5.parameters(), lr=learning_rate, weight_decay=weight_decay)
        name = 'model'+str(i)+"/"
        create_folder(name)
        file=open('./model_sava/'+name+'r.txt','a')
        file.write(
            f"learning rate={learning_rate} | fc dropout={fc_dropout_rate} | lstm dropout={lstm_dropout_rate} | weight decay={weight_decay}")
        if os.path.exists('./model_sava/'+name+'model1.pth'):
            model_load(name)

        # train
        number_of_epoch = 6
        time_start = time.time()
        train(model5, train_loader, val_loader, loss_fn,
              optimizer5,
              number_of_epoch,i,file)

        i += 1
        time_end = time.time()
        print('time cost', (time_end - time_start) / 60, 'min')
        print(
            f"learning rate={learning_rate} | fc dropout={fc_dropout_rate} | lstm dropout={lstm_dropout_rate} | weight decay={weight_decay}")

        file.close

        input('将模型放入对应文件夹后按回车继续...\n')


