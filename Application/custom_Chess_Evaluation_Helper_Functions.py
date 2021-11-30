import numpy as np
import torch
from minimax_Helper_Functions import *

#################################################
# Custom Chess Evaluation Functions
#################################################

    #################################################
    # FEN Functions
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
        row, col = fencoordinateToModel(square)
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

def fencoordinateToModel(coordinate):
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
    # Board Dictionary to FEN Functions
    #################################################

def ensureCurrentStateVariables(app):
    app.stateVariables = [app.whiteKingMoved, app.blackKingMoved, app.blackRookKingSideMoved,
                        app.blackRookQueenSideMoved, app.whiteRookKingSideMoved, 
                        app.whiteRookQueenSideMoved, app.whiteCastled, app.blackCastled,
                        app.lastPiecePlayedOldCoordinate, app.lastPiecePlayedNewCoordinate,
                        app.lastPiecePlayed]

def createPieceStringDict():
    result = dict()
    result['WhitePawn'] = 'P'
    result['WhiteKnight'] = 'N'
    result['WhiteBishop'] = 'B'
    result['WhiteRook'] = 'R'
    result['WhiteQueen'] = 'Q'
    result['WhiteKing'] = 'K'
    result['BlackPawn'] = 'p'
    result['BlackKnight'] = 'n'
    result['BlackBishop'] = 'b'
    result['BlackRook'] = 'r'
    result['BlackQueen'] = 'q'
    result['BlackKing'] = 'k'
    return result

def createBoardString(board):
    numbers = [8, 7, 6, 5, 4, 3, 2, 1]
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    pieceStringDict = createPieceStringDict()
    boardString = ''
    for i in range(len(numbers)):
        emptySquares = 0
        for j in range(len(letters)):
            coordinate = letters[j] + str(numbers[i])
            piece = board[coordinate]
            if emptySquares == 0:
                if piece != None:
                    pieceString = piece.getSide() + piece.getName()
                    pieceString = pieceStringDict[pieceString]
                    boardString += pieceString
                else:
                    emptySquares += 1
                    if j == len(letters) - 1:
                        boardString += str(emptySquares)
            else:
                if piece != None:
                    boardString += str(emptySquares)
                    emptySquares = 0
                    pieceString = piece.getSide() + piece.getName()
                    pieceString = pieceStringDict[pieceString]
                    boardString += pieceString
                else:
                    emptySquares += 1
                    if j == len(letters) - 1:
                        boardString += str(emptySquares)
        boardString += '/'
    boardString = boardString[:len(boardString) - 1]
    return boardString

def createSideString(side):
    if side == 'White':
        sideString = 'w'
    else:
        sideString = 'b'
    return sideString

def createCastlingString(stateVariables):
    whiteKingMoved = stateVariables[0]
    blackKingMoved = stateVariables[1]
    blackRookKingSideMoved = stateVariables[2]
    blackRookQueenSideMoved = stateVariables[3]
    whiteRookKingSideMoved = stateVariables[4]
    whiteRookQueenSideMoved = stateVariables[5]
    whiteCastled = stateVariables[6]
    blackCastled = stateVariables[7]
    whiteString = ''
    blackString = ''
    if whiteKingMoved != True and whiteCastled == False:
        if whiteRookKingSideMoved != True:
            whiteString += 'K'
        if whiteRookQueenSideMoved != True:
            whiteString += 'Q'
    if blackKingMoved != True and blackCastled == False:
        if blackRookKingSideMoved != True:
            blackString += 'k'
        if blackRookQueenSideMoved != True:
            blackString += 'q'
    castlingString = whiteString + blackString
    if len(castlingString) == 0:
        return '-'
    else: return castlingString

def createEnpassantString(stateVariables):
    lastPiecePlayedOldCoordinate = stateVariables[8]
    lastPiecePlayedNewCoordinate = stateVariables[9]
    lastPiecePlayed = stateVariables[10]
    enpassantString = ''
    if lastPiecePlayed != None and lastPiecePlayed.getName() == 'Pawn':
        coordinateNum1 = int(lastPiecePlayedOldCoordinate[1])
        coordinateNum2 = int(lastPiecePlayedNewCoordinate[1])
        if (abs(coordinateNum2 - coordinateNum1) == 2):
            letter = lastPiecePlayedOldCoordinate[0]
            number = str((coordinateNum1 + coordinateNum2) // 2)
            enpassantSquare = letter + number
            enpassantString += enpassantSquare
            return enpassantString
    return '-'

def dictToFen(board, side, stateVariables):
    boardString = createBoardString(board)
    sideString = createSideString(side)
    castlingString = createCastlingString(stateVariables)
    enpassantString = createEnpassantString(stateVariables)
    fen = boardString + ' ' + sideString + ' ' + castlingString + ' ' + enpassantString
    return fen

    #################################################
    # ML Evaluation Model Functions
    #################################################

def alternateOneHotEncodingMethod(l):
  encoding = list()
  for i in range(len(l)):
    for j in range(len(l[0])):
      index = l[i][j]
      encodedList = [0] * 17
      encodedList[index] = 1
      encoding.append(encodedList)
  return encoding

def createTensor(l):
  numpyArray = np.array(l, dtype=np.float32)
  nonFlattenedTensor = torch.from_numpy(numpyArray)
  return nonFlattenedTensor

def createFeatureTensor(l):
  nonFlattenedTensor = createTensor(l)
  flattenedTensor = nonFlattenedTensor.view(64 * 17) # Total elements in the board array
  return flattenedTensor

def createEvaluationInput(app):
    board = app.boardDict
    side = app.sideToMove
    stateVariables = app.stateVariables
    fen = dictToFen(board, side, stateVariables)
    boardRepresentation = fenToBoard(fen)
    encoding = alternateOneHotEncodingMethod(boardRepresentation)
    featureTensor = createFeatureTensor(encoding)
    return featureTensor

def generateEvaluation(app, featureTensor):
    board = app.boardDict
    loaded_model = app.loaded_model
    output = loaded_model(featureTensor)
    rawEvaluation = output.detach().numpy()
    nonMaterialEvaluation = np.tan(rawEvaluation[0])
    nonMaterialEvaluation = round(nonMaterialEvaluation, 2)

    materialEvaluation = evaluatePosition(board)
    if materialEvaluation == 0.0:
        return nonMaterialEvaluation
    else:    
        return materialEvaluation
