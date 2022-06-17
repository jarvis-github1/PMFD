import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader,random_split
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import PIL
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKING_PATH="./"
WORKING_PATH1='D:/1A文件资料/代码/fake_bert_pre_train_model/'
test_WORKING_PATH='D:/1A文件资料/代码/fake dataset/weibo/'
TEXT_LENGTH=75
TEXT_HIDDEN=256
"""
read text file, find corresponding image path
"""

def load_data():
	data_set=dict()
	for dataset in ['train']:
		file=open(os.path.join(WORKING_PATH,"text_data/",dataset+".txt"),"r")
		for line in file:
			content=eval(line)
			image=content[0]
			sentence=content[1]
			group=content[2]
			if os.path.isfile(os.path.join(WORKING_PATH,"image_data/",image+".jpg")):
				if image in data_set:
					print(image)
				data_set[int(image)]={"text":sentence,"group":group}

	return data_set
# def test_load_data():
# 	data_set=dict()
# 	for dataset in ['test']:
# 		file=open(os.path.join(test_WORKING_PATH,"text_data/",dataset+".txt"),"r")
# 		for line in file:
# 			content=eval(line)
# 			image=content[0]
# 			sentence=content[1]
# 			group=content[2]
# 			if os.path.isfile(os.path.join(test_WORKING_PATH,"image_data/",image+".jpg")):
# 				if image in data_set:
# 					print(image)
# 				data_set[int(image)]={"text":sentence,"group":group}
#
# 	return data_set
data_set=load_data()     #{"id":{text:"sentence","group":group}}
# test_data_set=load_data()     #{"id":{text:"sentence","group":group}}

# load word index
def load_word_index():
	with open(os.path.join(WORKING_PATH,"text_embedding/vocabs.txt"), 'r',encoding='utf-8') as f:
		for line in f:
			word2index=eval(line)
	return word2index
word2index=load_word_index()   #{'word':index}   count:59241


class my_data_set(Dataset):
	def __init__(self, data):
		self.data=data
		self.image_ids=list(data.keys())
		for id in data.keys():
			self.data[id]["image_path"] = os.path.join(WORKING_PATH,"image_dat/",str(id)+".jpg")

        # load all text     
		for id in data.keys():
			text=self.data[id]["text"].split()             
			text_index=torch.empty(TEXT_LENGTH,dtype=torch.long)
			curr_length=len(text)
			for i in range(TEXT_LENGTH):
				if i>=curr_length:
					text_index[i]=word2index["<pad>"]
				elif text[i] in word2index:
					text_index[i]=word2index[text[i]]   
				else:
					text_index[i]=word2index["<pad>"]
			# print(text_index)
			self.data[id]["text_index"] = text_index

    # load image feature data - resnet 50 result
	def __image_feature_loader(self,id):
		image_feature = np.load(os.path.join(WORKING_PATH,"image_feature_data",str(id)+".npy"))
		return torch.from_numpy(image_feature)

	# load attribute feature data - 5 words label


	def __text_index_loader(self,id):
		return self.data[id]["text_index"]


	def image_loader(self,id):
		path=self.data[id]["image_path"]
		img_pil =  PIL.Image.open(path)
		transform = transforms.Compose([transforms.Resize((448,448)),
                                        transforms.ToTensor(),
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
		img_tensor = transform(img_pil)
		return img_tensor
	def text_loader(self,id):
		return self.data[id]["text"]

	def __getitem__(self, index):
		id=self.image_ids[index]
		# img = self.__image_loader(id)
		text_index = self.__text_index_loader(id)
		image_feature = self.__image_feature_loader(id)

		group = self.data[id]["group"]
		return text_index,image_feature,group,id

	def __len__(self):
		return len(self.image_ids)
def train_val_test_split(all_Data,train_fraction,val_fraction):
    # split the data
    train_val_test_count=[int(len(all_Data)*train_fraction),int(len(all_Data)*val_fraction),0]
    train_val_test_count[2]=len(all_Data)-sum(train_val_test_count)
    return random_split(all_Data,train_val_test_count,generator=torch.Generator().manual_seed(43))


all_Data=my_data_set(data_set)
train_fraction=0.003
val_fraction=0.003
batch_size=32 #32
train_set,val_set,test_set=train_val_test_split(all_Data,train_fraction,val_fraction)
# add to dataloader
# all_loader = DataLoader(all_Data,batch_size=batch_size, shuffle=True)

train_loader = DataLoader(train_set,batch_size=32, shuffle=False)

if __name__ == "__main__":
    for text_index,image_feature,group,id in train_loader:
        # plt.imshow(img[0].permute(1,2,0))
        # plt.show()
        print("text：",text_index.shape,text_index.type())#torch.Size([32, 75]) torch.LongTensor
        print("image feature：",image_feature.shape,image_feature.type())#torch.Size([32, 196, 2048]) torch.FloatTensor
        print("group：",group,group.type())#tensor([1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1,0, 1, 1, 1, 0, 0, 0, 1]) torch.LongTensor
        # print("image id：",id,id.type())
        break