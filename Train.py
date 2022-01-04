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


class imagemodel(torch.nn.Module):
    def __init__(self, fc_dropout_rate):
        super(imagemodel, self).__init__()
        self.image = ImageFeature.ExtractImageFeature()
        self.fuse = FuseAllFeature.ModalityFusion_1(1024)
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate, 512)

    def forward(self, image_feature):
        image_result, image_seq = self.image(image_feature)
        fusion = self.fuse(image_result, image_seq)
        output = self.final_classifier(fusion)
        return output


class textmodel(torch.nn.Module):
    def __init__(self, lstm_dropout_rate, fc_dropout_rate):
        super(textmodel, self).__init__()
        self.text = TextFeature.ExtractTextFeature(TEXT_LENGTH, TEXT_HIDDEN, lstm_dropout_rate)
        self.fuse = FuseAllFeature.ModalityFusion_1(512)
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate, 512)

    def forward(self, text_index):
        text_result, text_seq = self.text(text_index, None)
        fusion = self.fuse(text_result, text_seq.permute(1, 0, 2))
        output = self.final_classifier(fusion)
        return output



class Multimodel_image_text(torch.nn.Module):
    def __init__(self, lstm_dropout_rate, fc_dropout_rate):
        super(Multimodel_image_text, self).__init__()
        self.image = ImageFeature.ExtractImageFeature()
        self.text = TextFeature.ExtractTextFeature(TEXT_LENGTH, TEXT_HIDDEN, lstm_dropout_rate)
        self.fuse = FuseAllFeature.ModalityFusion_2(1024, 512)
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate, 512)

    def forward(self, image_feature, text_index):
        image_result, image_seq = self.image(image_feature)
        text_result, text_seq = self.text(text_index, None)
        fusion = self.fuse(image_result, image_seq, text_result, text_seq.permute(1, 0, 2))
        output = self.final_classifier(fusion)
        return output

def train(model1, model2, model3, train_loader, val_loader, loss_fn, optimizer1,
          optimizer2, optimizer3 , number_of_epoch,i):
    F1_old = 0
    TP=TN=FN=FP=0
    F1=0
    TP1 = TN1 = FN1 = FP1 = 0
    TP2 = TN2 = FN2 = FP2 = 0
    TP3 = TN3 = FN3 = FP3 = 0
    p1=p2=p3=r1=r2=r3=0
    for epoch in range(number_of_epoch):

        model1_train_loss = 0
        model2_train_loss = 0
        model3_train_loss = 0

        cla_train_loss = 0
        cla_correct_train = 0
        # for i in range(1,9):
        #     model_train_loss[i]=0
        #     model_correct_train[i] = 0
        #     eval('model'+str(i)+'.train()')
        model1_correct_train = 0
        model2_correct_train = 0
        model3_correct_train = 0

        model1.train()
        model2.train()
        model3.train()


        right_num = data_num = count = 0
        dict={}
        for text_index, image_feature, group, id in train_loader:
            count += 1
            dataset_len = len(train_loader)
            data_num += train_loader.batch_size
            group = group.view(-1, 1).to(torch.float32).to(device)
            model1_pred = model1(image_feature.to(device))
            model2_pred = model2(text_index.to(device))
            model3_pred = model3(image_feature.to(device), text_index.to(device))

            # result  = torch.cat((model1_pred,model2_pred,model3_pred,model4_pred,model5_pred,model6_pred,model7_pred,model8_pred),1)
            # result = fal(result)
            # cla_loss = loss_fn(result,group)
            # cla_train_loss += cla_loss
            # cla_correct_train += (result.round() == group).sum().item()
            # optimizer.zero_grad()
            # cla_loss.backward(retain_graph=True)
            # print(result)
            # optimizer.step()
            # # result = torch.cat((eval('model'+str(dict[0][0][0])+'_pred'), eval('model'+str(dict[0][1][0])+'_pred'), eval('model'+str(dict[0][2][0])+'_pred'), eval('model'+str(dict[0][3][0])+'_pred'),eval('model'+str(dict[0][4][0])+'_pred')), 1)
            # # result = torch.mean(result, dim=1, keepdim=True)
            # right_num += (result.round() == group).sum().item()

            model1_loss = loss_fn(model1_pred, group)
            model2_loss = loss_fn(model2_pred, group)
            model3_loss = loss_fn(model3_pred, group)

            model1_train_loss += model1_loss
            model2_train_loss += model2_loss
            model3_train_loss += model3_loss

            model1_correct_train += (model1_pred.round() == group).sum().item()
            model2_correct_train += (model2_pred.round() == group).sum().item()
            model3_correct_train += (model3_pred.round() == group).sum().item()

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            model1_loss.backward()
            model2_loss.backward()
            model3_loss.backward()

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            # TP += ((model3_pred.round() == 1) & (group == 1)).cpu().sum().item()
            # TN += ((model3_pred.round() == 0) & (group == 0)).cpu().sum().item()
            # FN += ((model3_pred.round() == 0) & (group == 1)).cpu().sum().item()
            # FP += ((model3_pred.round() == 1) & (group == 0)).cpu().sum().item()
            # def F1_score(p,r):
            #     F1=2 * r * p / (r + p)
            #     return F1
            def f1_score(TP,TN,FN,FP,model):
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
                return TP,TN,FN,FP,p,r,F1
            try:
                TP1, TN1, FN1, FP1,p1,r1,F1_1 = f1_score(TP1, TN1, FN1, FP1, model1_pred)
                TP2, TN2, FN2, FP2,p2,r2, F1_2 = f1_score(TP2, TN2, FN2, FP2, model2_pred)
                TP3, TN3, FN3, FP3,p3,r3, F1_3 = f1_score(TP3, TN3, FN3, FP3, model3_pred)
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
                    "pic:train_loss=%.5f train_acc=%.3f precision=%.5f recall=%.5f F1_score=%.5f" % (model1_train_loss / data_num, model1_correct_train / data_num,p1,r1,F1_1))
                print(
                    "txt:train_loss=%.5f train_acc=%.3f precision=%.5f recall=%.5f F1_score=%.5f" % (model2_train_loss / data_num, model2_correct_train / data_num,p2,r2,F1_2))
                print(
                    "pic_txt:train_loss=%.5f train_acc=%.3f precision=%.5f recall=%.5f F1_score=%.5f" % (model3_train_loss / data_num, model3_correct_train / data_num,p3,r3,F1_3))
                print('')
            except:
                print('Training...epoch:%d/%d' % (epoch + 1, number_of_epoch))
                print('batch:%d/%d' % (count, dataset_len))
                print("pic:train_loss=%.5f train_acc=%.3f " % (model1_train_loss / data_num, model1_correct_train / data_num))
                print("txt:train_loss=%.5f train_acc=%.3f " % (model2_train_loss / data_num, model2_correct_train / data_num))
                print("pic_txt:train_loss=%.5f train_acc=%.3f " % (model3_train_loss / data_num, model3_correct_train / data_num))
                print('')
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
        F1_new = test(model1, model2, model3, val_loader)
        
        # test(model1, model2, model3,  val_loader)
        if F1_new < F1_old:
            print('Learning rate changed')
            optimizer1.param_groups[0]['lr'] *= 0.8
            optimizer2.param_groups[0]['lr'] *= 0.8
            optimizer3.param_groups[0]['lr'] *= 0.8


        F1_old = F1_new

        print('learning_rate:', optimizer3.param_groups[0]['lr'])
        print('F1:', F1_new)
        print('')

        # sava model
        state1 = {'model': model1.state_dict(), 'optimizer': optimizer1.state_dict()}
        state2 = {'model': model2.state_dict(), 'optimizer': optimizer2.state_dict()}
        state3 = {'model': model3.state_dict(), 'optimizer': optimizer3.state_dict()}

        name='model'+str(i)+"/"
        torch.save(state1, './model_sava/'+name+'model1.pth')
        torch.save(state2, './model_sava/'+name+'model2.pth')
        torch.save(state3, './model_sava/'+name+'model3.pth')



def test(model1, model2, model3,val_loader):
    valid_loss1 = 0
    valid_loss2 = 0
    valid_loss3 = 0

    correct_valid1 = 0
    correct_valid2 = 0
    correct_valid3 = 0

    model1.eval()
    model2.eval()
    model3.eval()

    right_num = data_num = count = 0
    TP = TN = FP = FN = 0
    F1=0
    with torch.no_grad():
        for val_text_index, val_image_feature, val_grad, val_id in val_loader:
            count += 1
            dataset_len = len(val_loader)
            val_group = val_grad.view(-1, 1).to(torch.float32).to(device)
            model1_val_pred = model1(val_image_feature.to(device))
            model2_val_pred = model2(val_text_index.to(device))
            model3_val_pred = model3(val_image_feature.to(device),val_text_index.to(device))

            # right_num += (result.round() == val_group).sum().item()
            data_num += val_loader.batch_size
            val_loss1 = loss_fn(model1_val_pred, val_group)
            val_loss2 = loss_fn(model2_val_pred, val_group)
            val_loss3 = loss_fn(model3_val_pred, val_group)

            valid_loss1 += val_loss1
            valid_loss2 += val_loss2
            valid_loss3 += val_loss3

            correct_valid1 += (model1_val_pred.round() == val_group).sum().item()
            correct_valid2 += (model2_val_pred.round() == val_group).sum().item()
            correct_valid3 += (model3_val_pred.round() == val_group).sum().item()


            TP += ((model3_val_pred.round() == 1) & (val_group == 1)).cpu().sum().item()
            TN += ((model3_val_pred.round() == 0) & (val_group == 0)).cpu().sum().item()
            FN += ((model3_val_pred.round() == 0) & (val_group == 1)).cpu().sum().item()
            FP += ((model3_val_pred.round() == 1) & (val_group == 0)).cpu().sum().item()
            # compute F1
            p = TP / (TP + FP)
            # p = torch.floor_divide(TP,TP+FP)
            r = TP / (TP + FN)
            # r = torch.floor_divide(TP, TP + FN)
            F1 = 2 * r * p / (r + p)
            print('Validing...')
            print('batch:%d/%d' % (count, dataset_len))
            print("pic:valid_loss=%.5f valid_acc=%.3f" % (valid_loss1 / data_num, correct_valid1 / data_num))
            print("txt:valid_loss=%.5f valid_acc=%.3f" % (valid_loss2 / data_num, correct_valid2 / data_num))
            print("pic_txt:valid_loss=%.5f valid_acc=%.3f F1_score=%.5f" % (valid_loss3 / data_num, correct_valid3 / data_num,F1))

            print('')


        return F1


def model_load(name):
    checkpoint1 = torch.load('./model_sava/'+name+'model1.pth')
    checkpoint2 = torch.load('./model_sava/'+name+'model2.pth')
    checkpoint3 = torch.load('./model_sava/'+name+'model3.pth')
    # try:
    #     model1.load_state_dict(checkpoint1['model'])
    #     model2.load_state_dict(checkpoint2['model'])
    #     model3.load_state_dict(checkpoint3['model'])
    # except:
    #     pass
    # model1.load_state_dict(checkpoint1['model'])
    model1.load_state_dict(checkpoint1['model'])
    model2.load_state_dict(checkpoint2['model'])
    model3.load_state_dict(checkpoint3['model'])

    optimizer1.load_state_dict(checkpoint1['optimizer'])
    optimizer2.load_state_dict(checkpoint2['optimizer'])
    optimizer3.load_state_dict(checkpoint3['optimizer'])

    print('Model_Load completed\n')
    return model1, model2, model3, optimizer1, optimizer2, optimizer3


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
    i =7
    for learning_rate, fc_dropout_rate, lstm_dropout_rate, weight_decay in list(comb)[7:]:
        print(
            f"learning rate={learning_rate} | fc dropout={fc_dropout_rate} | lstm dropout={lstm_dropout_rate} | weight decay={weight_decay}")
        # loss function
        loss_fn = torch.nn.BCELoss()
        # initilize the model
        model1 = imagemodel(fc_dropout_rate).to(device)
        model2 = textmodel(lstm_dropout_rate, fc_dropout_rate).to(device)
        model3= Multimodel_image_text(lstm_dropout_rate, fc_dropout_rate).to(device)

        # optimizer
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer3 = torch.optim.Adam(model3.parameters(), lr=learning_rate, weight_decay=weight_decay)

        name = 'model'+str(i)+"/"
        create_folder(name)
        if os.path.exists('./model_sava/'+name+'model1.pth'):
            model_load(name)

        # train
        number_of_epoch = 4
        time_start = time.time()
        train(model1, model2, model3, train_loader, val_loader, loss_fn,
              optimizer1, optimizer2, optimizer3,
              number_of_epoch,i)
        i += 1
        time_end = time.time()
        print('time cost', (time_end - time_start) / 60, 'min')
        print(
            f"learning rate={learning_rate} | fc dropout={fc_dropout_rate} | lstm dropout={lstm_dropout_rate} | weight decay={weight_decay}")
        # input('将模型放入对应文件夹后按回车继续...\n')


