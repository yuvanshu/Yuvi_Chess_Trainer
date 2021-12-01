import numpy as np
import torch
import joblib
import pandas

#################################################
# Loading Chess Evaluation Dataset
#################################################

# Citation for Dataset: https://www.kaggle.com/ronakbadhe/chess-evaluations
# This source details where the data was found and downloaded from

# Citation: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# This documentation explains how to load a csv file using pandas.read_csv() function

# Specify the directory that the chess evaluation data csv is stored
data_csv_path = 'Your Directory Path Goes Here'

# nrows specified to 500000 since model will be trained on 500000 positions
# If nrows > 500000, model may crash due to computation limitations
data_csv = pandas.read_csv(data_csv_path, nrows = 500000)

#################################################
# FEN To Board Representation Functions
#################################################

def fenToBoard(fen):
    board = [[0] * 8 for _ in range(8)]
    encodedValuesDict = createEncodeValuesDict()
    pieceSet = {'p', 'b', 'n', 'r', 'q', 'k', 'P', 'B', 'N', 'R', 'Q', 'K'}
    partIndex = 0
    for part in fen.split(' '):
        if partIndex == 0:
            boardString = part
            partIndex += 1
        elif partIndex == 1:
            side = part
            partIndex += 1
        elif partIndex == 2:
            castlingOptions = part
            partIndex += 1
        elif partIndex == 3:
            enpassantSquare = part
            partIndex += 1
    k, q, K, Q = castlingStringToBool(castlingOptions)

    rowIndex = 0
    for row in boardString.split('/'):
        colIndex = 0
        stringIndex = 0
        while (colIndex < len(board[0])):
            value = row[stringIndex]
            if value in pieceSet:
                if value == 'r' or value == 'R':
                    board = updateBoardForRooks(board, value, k, q, K, Q, encodedValuesDict, rowIndex, colIndex)
                    colIndex += 1
                    stringIndex += 1
                elif value == 'k' or value == 'K':
                    board = updateBoardForKings(board, value, side, rowIndex, colIndex)
                    colIndex += 1
                    stringIndex += 1
                elif value == 'p' or value == 'P':
                    board = updateBoardForPawns(board, value, enpassantSquare, side, encodedValuesDict, rowIndex, colIndex)
                    colIndex += 1
                    stringIndex += 1
                else: 
                    board = updateBoardRegularPiece(board, value, encodedValuesDict, rowIndex, colIndex)
                    colIndex += 1
                    stringIndex += 1
            else:
                board = updateBoardEmptySpaces(board, value, rowIndex, colIndex)
                colIndex += int(value)
                stringIndex += 1

        rowIndex += 1
    return board

def castlingStringToBool(s):
    if s.find('k') != -1: k = True
    else: k = False
    if s.find('q') != -1: q = True
    else: q = False
    if s.find('K') != -1: K = True
    else: K = False
    if s.find('Q') != -1: Q = True
    else: Q = False
    return k, q, K, Q

def updateBoardForRooks(board, val, k, q, K, Q, d, r, c):
    if (r,c) not in {(0,0), (0,7), (7,0), (7,7)}:
        board[r][c] = d[val]
    else:
        if (r,c) == (0,0): 
            if q == True:
                board[r][c] = 12 # black rook that can castle
            else: board[r][c] = d[val]
        elif (r,c) == (0,7):
            if k == True:
                board[r][c] = 12 # black rook that can castle
            else: board[r][c] = d[val]
        elif (r,c) == (7,0):
            if Q == True:
                board[r][c] = 11 # white rook that can castle
            else: board[r][c] = d[val]
        else:
            if K == True:
                board[r][c] = 11 # white rook that can castle
            else: board[r][c] = d[val]
    return board

def updateBoardForKings(board, val, side, r, c):
    if side == 'w' and val == 'K':
        board[r][c] = 13 # white king if white has the move
    elif side == 'b' and val == 'k':
        board[r][c] = 14 # black king if black has the move
    else:
        board[r][c] = 15 # the king that doesn't have the move
    return board


def updateBoardForPawns(board, val, square, side, d, r, c):
    if square != '-':
        row, col = coordinateToModel(square)
        if side == 'w':
            row, col = row + 1, col
        else:
            row, col = row - 1, col
        if r == row and c == col:
            board[r][c] = 16 # a pawn that has just advanced two squares
        else:
            board[r][c] = d[val]
    else:
        board[r][c] = d[val]
    return board
            
def updateBoardRegularPiece(board, val, d, r, c):
    board[r][c] = d[val]
    return board

def updateBoardEmptySpaces(board, val, r, c):
    emptySpaces = int(val)
    for space in range(emptySpaces):
        board[r][c + space] = 0 # empty square
    return board

def coordinateToModel(coordinate):
    coordinateDict = dict()
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['8', '7', '6', '5', '4', '3', '2', '1']
    index = 0
    for letter in letters:
        coordinateDict[letter] = index
        index += 1
    index = 0
    for number in numbers:
        coordinateDict[number] = index
        index += 1
    row = coordinateDict[coordinate[1]]
    col = coordinateDict[coordinate[0]]
    return row, col

def createEncodeValuesDict():
    encodedValuesDict = dict()
    pieces = ['P', 'p', 'N', 'n', 'B', 'b', 'Q', 'q', 'R', 'r']
    index = 1
    for piece in pieces:
        encodedValuesDict[piece] = index
        index += 1
    return encodedValuesDict


#################################################
# Features and Labels Helper Functions
#################################################

# Creates a tensor from given list
def createTensor(l):
    # Citation: https://www.aiworkbox.com/lessons/convert-a-numpy-array-to-a-pytorch-tensor
    # Code explains how to create a numpy array from a list and convert numpy array to tensor

    numpyArray = np.array(l, dtype=np.float32)
    nonFlattenedTensor = torch.from_numpy(numpyArray)
    return nonFlattenedTensor

def createFeatureTensor(l):
    # Citation: https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
    # Documentation explains how Tensor.view() function works as implemented here
    # Refer to citation for all use cases of Tensor.view() function

    nonFlattenedTensor = createTensor(l)
    flattenedTensor = nonFlattenedTensor.view(64 * 17) # Total elements in the board array
    return flattenedTensor

def readableEvaluationValues(s):
    # s represents a string evaluation

    # evaluation is not mate
    if s[0] != '#':
        if s[0] == '0':
            return int(s[0])
        else:    
            value  = int(s[1:]) / 100
            sign = s[0]
            if sign == '+':
                return value 
            else: return value * -1

    # creating a numerical value for mate based off # moves till mate
    else:
        allPieceValues = 3900 # in centi pawns
        movesToMate = int(s[2:])
        if movesToMate == 0:
          return 0
        sign = s[1]
        maxMateDepth = 25
        value = ((allPieceValues * maxMateDepth) / movesToMate) / 100
        return value

def alternateOneHotEncodingMethod(l):
    encoding = list()
    for i in range(len(l)):
        for j in range(len(l[0])):
            index = l[i][j]
            encodedList = [0] * 17
            encodedList[index] = 1
            encoding.append(encodedList)
    return encoding

def createLabelTensor(val):
  labelList = list()
  labelList.append(val)
  nonFlattenedTensor = createTensor(labelList)
  flattenedTensor = nonFlattenedTensor.view(1) 
  return flattenedTensor

def createFeatures(df):
  features = []
  # Feature index is 0
  for i in range(len(df)):
    fen = df.iloc[i, 0]
    representation = fenToBoard(fen)
    encoding = alternateOneHotEncodingMethod(representation)
    featureTensor = createFeatureTensor(encoding)
    features.append(featureTensor)
  return features

def createLabels(df):
  labels = []
  # Label index is 1
  for i in range(len(df)):
    evaluation = df.iloc[i, 1]
    new_evaluation = readableEvaluationValues(evaluation)
    arctan_evaluation = np.arctan(new_evaluation)
    labelTensor = createLabelTensor(arctan_evaluation)
    labels.append(labelTensor)
  return labels

#################################################
# Creating features and labels
#################################################

# Citation: https://www.kaggle.com/ferrangh95/initial-training-feed-forward-net
# Note: All code within this section collected from citation source

features = createFeatures(data_csv)
labels = createLabels(data_csv)

#################################################
# Creating Training and Testing Variables
#################################################

# Citation: https://www.kaggle.com/ferrangh95/initial-training-feed-forward-net
# Note: All code within this section collected from citation source

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42, shuffle=True)

#################################################
# Creating Chess Dataset Class
#################################################

# Citation: https://www.kaggle.com/ferrangh95/initial-training-feed-forward-net
# Note: All code within this section collected from citation source

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ChessDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        item = dict(features=self.features[idx], labels=self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

#################################################
# Creating Training and Testing Dataset
#################################################

# Citation: https://www.kaggle.com/ferrangh95/initial-training-feed-forward-net
# Note: All code within this section collected from citation source

train_dataset = ChessDataset(features = X_train, labels=y_train)
test_dataset = ChessDataset(features = X_test, labels=y_test)

#################################################
# Creating Training and Testing Dataset
#################################################

# Citation: https://www.kaggle.com/ferrangh95/initial-training-feed-forward-net
# Note: All code within this section collected from citation source

# Empties caches preparing to create dataloaders
torch.cuda.empty_cache()

# Respective batch size values were determined by me. Differ from those used in citation
train_batch_size = 10000 
test_batch_size = 5000

training_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
testing_data_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

#################################################
# Setting Model Architecture
#################################################

# Citation: https://www.kaggle.com/ferrangh95/initial-training-feed-forward-net
# Note: All code within this section collected from citation source
# However, specifications of model architecture (such as how many activation functions,
# dropout layers, and their input size were come up by me)

from torch.nn.init import xavier_uniform_
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    try:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    except Exception:
        return
    
class MLP(torch.nn.Module):
    # n_inputs should be 1088. That is the size of the encoding
    def __init__(self, n_inputs):
        super(MLP, self).__init__()    
        self.base_model = torch.nn.Sequential(
        torch.nn.Linear(n_inputs, 544),
        torch.nn.ReLU(),
        torch.nn.Dropout(.2),
        torch.nn.Linear(544, 272),
        torch.nn.Dropout(.2),
        torch.nn.ReLU(),
        torch.nn.Linear(272, 136),
        torch.nn.Dropout(.2),
        torch.nn.ReLU(),
        torch.nn.Linear(136, 68),
        torch.nn.Dropout(.2),
        torch.nn.ReLU(),
        torch.nn.Linear(68, 1),
    ).to(device)
        self.base_model.apply(init_weights)

    # forward propagate input
    def forward(self, X):
        X = X.to(device)
        # input to first hidden layer
        X = self.base_model(X)
        return X

#################################################
# Running the model
#################################################

# Citation 1: https://www.kaggle.com/ferrangh95/initial-training-feed-forward-net
# Citation 2: https://www.youtube.com/watch?v=9j-_dOze4IM&list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh&index=4

import torch.optim as optim     # Citation 1

model = MLP(1088) # n_input size is 1088    # Citation 1
optimizer = optim.Adam(model.parameters(), lr = 0.001) # Citation 2
criterion = torch.nn.MSELoss() # Citation 1

epochs = 45 # This value was determined by me

for epoch in range(epochs):     # Citation 2
  for data in training_data_loader:     # Citation 2
    X = data['features']    # Citation 2
    y = data['labels']      # Citation 2
    model.zero_grad()       # Citation 2
    output = model(X)       # Citation 2
    y = y.to(device)        # Citation 1     
    loss = criterion(output, y)     # Citation 1
    loss.backward()     # Citation 2
    optimizer.step()    # Citation 2
  print(loss)

#################################################
# Saving the Model 
#################################################

# Citation: https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html
# This documentation explains how to save a model using joblib.dump() function

# Specify the directory to save the model to on your host computer
filepath = 'Your Directory Path Goes Here'

joblib.dump(model, filepath)

#################################################
# Loading the Model From Directory You Saved To
#################################################

# Citation: https://joblib.readthedocs.io/en/latest/generated/joblib.load.html
# This documentation explains how to load a model using joblib.load() function

loaded_model = joblib.load(filepath)
