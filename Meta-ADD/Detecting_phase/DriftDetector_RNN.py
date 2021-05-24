import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net50(nn.Module):
    """
    Image2Vector CNN which takes image of dimension (28x28x3) and return column vector length 64
    """

    def __init__(self):
        super(Net50, self).__init__()
        self.fc1 = nn.Linear(50, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 30)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = torch.flatten(x, start_dim=1)
        return x

class Net100(nn.Module):
    """
    Image2Vector CNN which takes image of dimension (28x28x3) and return column vector length 64
    """

    def __init__(self):
        super(Net100, self).__init__()
        self.fc1 = nn.Linear(100, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 250)
        self.fc4 = nn.Linear(250, 30)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = torch.flatten(x, start_dim=1)
        return x

class Net200(nn.Module):
    """
    Image2Vector CNN which takes image of dimension (28x28x3) and return column vector length 64
    """

    def __init__(self):
        super(Net200, self).__init__()
        self.fc1 = nn.Linear(200, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 30)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = torch.flatten(x, start_dim=1)
        return x

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.n_hidden = 64
        feature_len = 1
        self.encoder = nn.GRU(input_size=feature_len, hidden_size=self.n_hidden, num_layers=1, dropout=0.1)
        self.middle = nn.GRU(input_size=feature_len, hidden_size=self.n_hidden, num_layers=1, dropout=0.1)
        self.decoder = nn.GRU(input_size=feature_len, hidden_size=self.n_hidden, num_layers=1, dropout=0.1)
        self.fc = nn.Linear(self.n_hidden, 30)

    def forward(self, src_batch: torch.LongTensor):

        # src_batch: [seq_len, batch_size, feature_len] = [200, 20, 1]
        src_batch = src_batch.transpose(0, 1).unsqueeze(dim=2)
        seq_len, batch_size, feature_len = src_batch.shape

        enc_outputs, hidden_cell = self.encoder(src_batch, torch.zeros(1, batch_size, self.n_hidden))

        middle_input = torch.zeros(1, batch_size, 1)  # initialize the initial state
        middle_output, hidden_cell = self.middle(middle_input, hidden_cell)

        decoder_input = torch.zeros(1, batch_size, 1) # initialize the initial state
        decoder_output, hidden_cell = self.decoder(decoder_input, hidden_cell)

        outputs = self.fc(hidden_cell).squeeze(0)
        return outputs


def InputEmbeding(input, BASE_PATH, Data_Vector_Length, ModelSelect):
    PATH = BASE_PATH+'/model_embeding.pkl'

    if ModelSelect == 'RNN':
        model_embeding = RNN()
    elif ModelSelect == 'FCN':
        if Data_Vector_Length == 50:
            model_embeding = Net50()
        elif Data_Vector_Length == 100:
            model_embeding = Net100()
        else:
            model_embeding = Net200()
    model_embeding.load_state_dict(torch.load(PATH))
    return model_embeding(input)

def get_Input_Data(BASE_PATH):
    Test_Example_Label = 1.0
    DATA_PATH = BASE_PATH+'/Test_Example_Data.pt'
    Test_Example_Data = torch.load(DATA_PATH)

    return Test_Example_Data

def Detector(Query_x, BASE_PATH, Data_Vector_Length, ModelSelect):
    DATA_PATH = BASE_PATH+'/centroid_matrix.pt'
    centroid_matrix = torch.load(DATA_PATH)

    Query_x = InputEmbeding(Query_x, BASE_PATH, Data_Vector_Length, ModelSelect)
    m = Query_x.size(0)
    n = centroid_matrix.size(0)
    centroid_matrix = centroid_matrix.expand(
        m, centroid_matrix.size(0), centroid_matrix.size(1))

    Query_matrix = Query_x.expand(n, Query_x.size(0), Query_x.size(
        1)).transpose(0, 1)  # Expanding Query matrix "n" times
    # Temp_A = centroid_matrix.transpose(1, 2)
    # Temp_B = Query_matrix.transpose(1, 2)
    Qx = torch.cosine_similarity(centroid_matrix.transpose(
        1, 2), Query_matrix.transpose(1, 2),dim=1)
    return Qx

def main(BASE_PATH, Data_Vector_Length, ModelSelect):
    Test_Example_Data = get_Input_Data(BASE_PATH)
    Qx = Detector(Test_Example_Data, BASE_PATH, Data_Vector_Length, ModelSelect)
    pred = torch.log_softmax(Qx, dim=-1)
    # DataType: float
    Label = float((torch.argmax(pred, 1)[0]))

    print(Label)

if __name__ == "__main__":
    # File address
    DATA_FILE = 'drift-200-25'

    # 50 OR 100 OR 200
    Data_Vector_Length = 200

    ModelSelect = 'RNN' # 'RNN', 'FCN', 'Seq2Seq'

    BASE_PATH = 'input/model/'+ ModelSelect+'/'+DATA_FILE

    main(BASE_PATH, Data_Vector_Length, ModelSelect)