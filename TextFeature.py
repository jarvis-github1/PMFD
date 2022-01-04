import torch
import numpy as np
import LoadData

class ExtractTextFeature(torch.nn.Module):
    def __init__(self,text_length,hidden_size,dropout_rate=0.2):
        super(ExtractTextFeature, self).__init__()
        self.hidden_size=hidden_size
        self.text_length=text_length
        embedding_weight=self.getEmbedding()
        self.embedding_size=embedding_weight.shape[1]
        self.embedding=torch.nn.Embedding.from_pretrained(embedding_weight)
        self.biLSTM=torch.nn.LSTM(input_size=200,hidden_size=hidden_size,bidirectional=True,batch_first=True)

        # early fusion
        self.Linear_1=torch.nn.Linear(200,hidden_size)
        self.Linear_2=torch.nn.Linear(200,hidden_size)
        self.Linear_3=torch.nn.Linear(200,hidden_size)
        self.Linear_4=torch.nn.Linear(200,hidden_size)

        # dropout
        self.dropout=torch.nn.Dropout(dropout_rate)
    def forward(self, input,guidence ):
        embedded=self.embedding(input).view(-1, self.text_length, self.embedding_size)

        if(guidence is not None):
            # early fusion
            hidden_init=torch.stack([torch.relu(self.Linear_1(guidence)),torch.relu(self.Linear_2(guidence))],dim=0)
            cell_init=torch.stack([torch.relu(self.Linear_3(guidence)),torch.relu(self.Linear_4(guidence))],dim=0)
            output,_=self.biLSTM(embedded,(hidden_init,cell_init))
        else:
            output,_=self.biLSTM(embedded,None)

        # dropout
        output=self.dropout(output)

        RNN_state=torch.mean(output,1)
        return RNN_state,output

    def getEmbedding(self):
        return torch.from_numpy(np.loadtxt("text_embedding/vectors.txt", delimiter=' ', dtype='float32'))

class ExtractTextFeature_lstm(torch.nn.Module):
    def __init__(self,text_length,hidden_size,dropout_rate=0.2):
        super(ExtractTextFeature_lstm, self).__init__()
        self.hidden_size=hidden_size
        self.text_length=text_length
        embedding_weight=self.getEmbedding()
        self.embedding_size=embedding_weight.shape[1]
        self.embedding=torch.nn.Embedding.from_pretrained(embedding_weight)
        self.biLSTM=torch.nn.LSTM(input_size=200,hidden_size=hidden_size,bidirectional=False,batch_first=True)

        # early fusion
        self.Linear_1=torch.nn.Linear(200,hidden_size)
        self.Linear_2=torch.nn.Linear(200,hidden_size)
        self.Linear_3=torch.nn.Linear(200,hidden_size)
        self.Linear_4=torch.nn.Linear(200,hidden_size)

        # dropout
        self.dropout=torch.nn.Dropout(dropout_rate)
    def forward(self, input,guidence ):
        embedded=self.embedding(input).view(-1, self.text_length, self.embedding_size)

        if(guidence is not None):
            # early fusion
            hidden_init=torch.stack([torch.relu(self.Linear_1(guidence)),torch.relu(self.Linear_2(guidence))],dim=0)
            cell_init=torch.stack([torch.relu(self.Linear_3(guidence)),torch.relu(self.Linear_4(guidence))],dim=0)
            output,_=self.biLSTM(embedded,(hidden_init,cell_init))
        else:
            output,_=self.biLSTM(embedded,None)

        # dropout
        output=self.dropout(output)

        RNN_state=torch.mean(output,1)
        return RNN_state,output

    def getEmbedding(self):
        return torch.from_numpy(np.loadtxt("text_embedding/vectors.txt", delimiter=' ', dtype='float32'))
class ExtractTextFeature_g(torch.nn.Module):
    def __init__(self,text_length,hidden_size,dropout_rate=0.2):
        super(ExtractTextFeature_g, self).__init__()
        self.hidden_size=hidden_size
        self.text_length=text_length
        embedding_weight=self.getEmbedding()
        self.embedding_size=embedding_weight.shape[1]
        self.embedding=torch.nn.Embedding.from_pretrained(embedding_weight)
        self.biLSTM=torch.nn.LSTM(input_size=200,hidden_size=hidden_size,bidirectional=True,batch_first=True)

        # early fusion
        self.Linear_1=torch.nn.Linear(1024,hidden_size)
        self.Linear_2=torch.nn.Linear(1024,hidden_size)
        self.Linear_3=torch.nn.Linear(1024,hidden_size)
        self.Linear_4=torch.nn.Linear(1024,hidden_size)

        # dropout
        self.dropout=torch.nn.Dropout(dropout_rate)
    def forward(self, input,guidence ):
        embedded=self.embedding(input).view(-1, self.text_length, self.embedding_size)

        if(guidence is not None):
            # early fusion
            hidden_init=torch.stack([torch.relu(self.Linear_1(guidence)),torch.relu(self.Linear_2(guidence))],dim=0)
            cell_init=torch.stack([torch.relu(self.Linear_3(guidence)),torch.relu(self.Linear_4(guidence))],dim=0)
            output,_=self.biLSTM(embedded,(hidden_init,cell_init))
        else:
            output,_=self.biLSTM(embedded,None)

        # dropout
        output=self.dropout(output)

        RNN_state=torch.mean(output,1)
        return RNN_state,output

    def getEmbedding(self):
        return torch.from_numpy(np.loadtxt("text_embedding/vectors.txt", delimiter=' ', dtype='float32'))
if __name__ == "__main__":
    test=ExtractTextFeature_lstm(LoadData.TEXT_LENGTH, LoadData.TEXT_HIDDEN)
    for text_index,image_feature,group,id in LoadData.train_loader:
        print(text_index.shape)
        result,seq=test(text_index,None)
        # [2, 512]
        print(result.shape)
        # print(result)
        # [2, 75, 512]
        print(seq.shape)
        # print(seq)
        break


