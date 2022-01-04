
import torch
import numpy as np
import LoadData
import TextFeature
import ImageFeature
import FuseAllFeature

#%%
#
# class ClassificationLayer(torch.nn.Module):
#     def __init__(self,dropout_rate=0):
#         super(ClassificationLayer, self).__init__()
#         self.Linear_1=torch.nn.Linear(512,256)
#         self.Linear_2=torch.nn.Linear(256,1)
#         self.dropout=torch.nn.Dropout(dropout_rate)
#
#     def forward(self,input):
#         hidden=self.Linear_1(input)
#         hidden=self.dropout(hidden)
#
#         output=torch.sigmoid(self.Linear_2(hidden))
#         return output
# class ClassificationLayer_2(torch.nn.Module):
#     def __init__(self,dropout_rate=0):
#         super(ClassificationLayer_2, self).__init__()
#         self.Linear_1=torch.nn.Linear(519,256)
#         self.Linear_2=torch.nn.Linear(256,1)
#         self.dropout=torch.nn.Dropout(dropout_rate)
#
#     def forward(self,input):
#         hidden=self.Linear_1(input)
#         hidden=self.dropout(hidden)
#
#         output=torch.sigmoid(self.Linear_2(hidden))
#         return output
class ClassificationLayer(torch.nn.Module):
    def __init__(self, dropout_rate=0, dim = 0):
        super(ClassificationLayer, self).__init__()
        self.Linear_1 = torch.nn.Linear(dim, 256)
        self.linear_2 = torch.nn.Linear(256,128)
        self.Linear_3 = torch.nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, input):
        hidden = self.Linear_1(input)
        hidden = self.dropout(hidden)
        hidden = torch.relu(self.linear_2(hidden))
        output = torch.sigmoid(self.Linear_3(hidden))
        return output
class ClassificationLayer_1(torch.nn.Module):
    def __init__(self, dropout_rate=0, dim = 0):
        super(ClassificationLayer_1, self).__init__()
        self.Linear_1 = torch.nn.Linear(dim, 128)
        # self.linear_2 = torch.nn.Linear(256,128)
        self.Linear_2= torch.nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, input):
        hidden = self.Linear_1(input)
        hidden = self.dropout(hidden)
        # hidden = torch.relu(self.linear_2(hidden))
        output = torch.sigmoid(self.Linear_2(hidden))
        return output
class finallClass(torch.nn.Module):
    def __init__(self,dim = 0):
        super(finallClass,self).__init__()
        self.linear_1 = torch.nn.Linear(dim,6)
        self.linear_2 = torch.nn.Linear(6, 4)
        self.linear_3 = torch.nn.Linear(4, 2)
        self.linear_4 = torch.nn.Linear(2, 1)
    def forward(self, input):
        hidden = self.linear_1(input)
        hidden = self.linear_2(hidden)
        hidden = self.linear_3(hidden)
        output = torch.sigmoid(self.linear_4(hidden))
        return output
if __name__ == "__main__":
    image=ImageFeature.ExtractImageFeature()
    text=TextFeature.ExtractTextFeature(LoadData.TEXT_LENGTH, LoadData.TEXT_HIDDEN)
    attribute=AttributeFeature.ExtractAttributeFeature()
    fuse=FuseAllFeature.ModalityFusion()
    final_classifier=ClassificationLayer()
    for text_index,image_feature,attribute_index,group,id in LoadData.train_loader:
        image_result,image_seq=image(image_feature)
        attribute_result,attribute_seq=attribute(attribute_index)
        text_result,text_seq=text(text_index,attribute_result)

        output_text_image=fuse(image_result,image_seq,text_result,text_seq.permute(1,0,2),attribute_result,attribute_seq.permute(1,0,2))
        output_text_attr=fuse(image_result,image_seq,text_result,text_seq.permute(1,0,2),attribute_result,attribute_seq.permute(1,0,2))
        output_image_attr=fuse(image_result,image_seq,text_result,text_seq.permute(1,0,2),attribute_result,attribute_seq.permute(1,0,2))
        print(output.shape)
        result=final_classifier(output)
        predict=torch.round(result)


        print(result.shape)#[32,1]
        print(result)
        print(predict)
        input()
