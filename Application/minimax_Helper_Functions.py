from chessPieceClass import *
from legal_Moves_Given_Checks import *
from promotion_Helper_Functions import *
import copy
import math

#################################################
# Mini-Max Functions
#################################################

def getMovesGivenChecks(app, piece, board):
    side = piece.getSide()
    name = piece.getName()

    if name == 'Pawn':
        pawnMoves = getPawnMoves(app, piece, side)
        return legalMovesGivenChecks(app, piece, pawnMoves, board)
    elif name == 'Knight':
        knightMoves = getKnightMoves(app, piece, side)
        return legalMovesGivenChecks(app, piece, knightMoves, board)
    elif name == 'Bishop':
        bishopMoves = getBishopMoves(app, piece, side)
        return legalMovesGivenChecks(app, piece, bishopMoves, board)
    elif name == 'Rook':
        rookMoves = getRookMoves(app, piece, side)
        return legalMovesGivenChecks(app, piece, rookMoves, board)
    elif name == 'Queen':
        queenMoves = getQueenMoves(app, piece, side)
        return legalMovesGivenChecks(app, piece, queenMoves, board)
    elif name == 'King':
        kingMoves = getKingMoves(app, piece, side)
        return legalMovesGivenChecks(app, piece, kingMoves, board)

def getCastleMove(side, direction):
    if direction == 'kingside' and side == 'White':
        move = [(side, 'King', 'e1', False), (side, 'King', 'e1', True),
                (side, 'Rook', 'h1', False), (side, 'Rook', 'f1', True)]
    elif direction == 'queenside' and side == 'White':
        move = [(side, 'King', 'e1', False), (side, 'King', 'c1', True),
                (side, 'Rook', 'a1', False), (side, 'Rook', 'd1', True)]
    elif direction == 'kingside' and side == 'Black':
        move = [(side, 'King', 'e8', False), (side, 'King', 'g8', True),
                (side, 'Rook', 'h8', False), (side, 'Rook', 'f8', True)]
    else:
        move = [(side, 'King', 'e8', False), (side, 'King', 'c8', True),
                (side, 'Rook', 'a8', False), (side, 'Rook', 'd8', True)]
    return move

def getPromotionMove(side, oldCoordinate, coordinate, promotionPiece):
    move = [(side, 'Pawn', oldCoordinate, False), (side, promotionPiece, coordinate, True)]
    return move

def getEnPassantMove(app, side, oldCoordinate, coordinate):
    if side == 'White':
        otherSide = 'Black'
    else: otherSide = 'White'
    otherPawnCoordinate = app.lastPiecePlayedNewCoordinate
    move = [(side, 'Pawn', oldCoordinate, False), (side, 'Pawn', coordinate, True),
            (otherSide, 'Pawn', otherPawnCoordinate, False)]
    return move

def getLegalMoves(app, side, board):
    legalMoves = []
    for coordinate in board:
        piece = board[coordinate]
        oldCoordinate = coordinate
        if piece != None and piece.getSide() == side:
            moveTuples = getMovesGivenChecks(app, piece, board)
            coordinates = getCoordinateList(app, moveTuples)

            # Iterating through all possible coordinates as moves
            for coordinate in coordinates:
                move = []
                #Updating the old square
                updateOne = (piece.getSide(), piece.getName(), oldCoordinate, False)
                move.append(updateOne)

                #Updating the new square 
                updateTwo = (piece.getSide(), piece.getName(), coordinate, True)
                move.append(updateTwo)
   
                legalMoves.append(move)

                # Checking for promotion moves
                promotionPieceSet = {'Knight', 'Bishop', 'Rook', 'Queen'}
                if piece.getName() == 'Pawn' and piecePromoted(piece, coordinate):
                    for promotionPiece in promotionPieceSet:
                        move = getPromotionMove(side, oldCoordinate, coordinate, promotionPiece)
                        legalMoves.append(move)
            
            # Checking for en passant moves
            if piece.getName() == 'Pawn' and canEnPassant(app, piece):
                (row, col) = getEnPassantSquare(app)
                coordinate = modelToCoordinate(app, row, col)
                move = getEnPassantMove(app, side, oldCoordinate, coordinate)
                legalMoves.append(move)

    # Add castling moves:
    castlingDirections = ['kingside', 'queenside']
    for direction in castlingDirections:
        if canCastle(app, side, direction):
            move = getCastleMove(side, direction)
            legalMoves.append(move)

    return legalMoves

def updateBoard(move, board):
    for update in move:
        (side, pieceName, coordinate, show) = update
        if show == True:
            piece = ChessPiece(side, pieceName)
            board[coordinate] = piece
        else:
            board[coordinate] = None
    return board

def updateStateVariables(app, move):
    for update in move:
        (side, pieceName, coordinate, show) = update

        # Checking if king moved and for which side
        if pieceName == 'King' and side == 'White':
            app.whiteKingMoved = True
        elif pieceName == 'King' and side == 'Black':
            app.blackKingMoved = True
    
        # Setting last piece moved variables
        if show == False:
            app.lastPiecePlayedOldCoordinate = coordinate
        else:
            app.lastPiecePlayedNewCoordinate = coordinate
            app.lastPiecePlayed = ChessPiece(side, pieceName)

    # Checking if castling occurred and for which side
    if (move == [('White', 'King', 'e1', False), ('White', 'King', 'e1', True),
                ('White', 'Rook', 'h1', False), ('White', 'Rook', 'f1', True)]):
        app.whiteCastled = True
        app.whiteRookKingSideMoved = True        
    elif (move == [('White', 'King', 'e1', False), ('White', 'King', 'c1', True),
                ('White', 'Rook', 'a1', False), ('White', 'Rook', 'd1', True)]):
        app.whiteCastled = True
        app.whiteRookQueenSideMoved = True
    elif (move == [('Black', 'King', 'e8', False), ('Black', 'King', 'g8', True),
                ('Black', 'Rook', 'h8', False), ('Black', 'Rook', 'f8', True)]):
        app.blackCastled = True
        app.blackRookKingSideMoved = True
    elif (move == [('Black', 'King', 'e8', False), ('Black', 'King', 'c8', True),
                ('Black', 'Rook', 'a8', False), ('Black', 'Rook', 'd8', True)]): 
        app.blackCastled = True
        app.blackRookQueenSideMoved = True
    
    # Checking which rook moved from starting square
    if ('White', 'Rook', 'h1', None) in move:
        app.whiteRookKingSideMoved = True
    elif ('White', 'Rook', 'a1', None) in move:
        app.whiteRookQueenSideMoved = True
    elif ('Black', 'Rook', 'h8', None) in move:
        app.blackRookKingSideMoved = True
    elif ('Black', 'Rook', 'a8', None) in move:
        app.blackRookQueenSideMoved = False
    return

def resetStateVariables(app, stateVariables):
    app.whiteKingMoved = stateVariables[0]
    app.blackKingMoved = stateVariables[1]
    app.blackRookKingSideMoved = stateVariables[2]
    app.blackRookQueenSideMoved = stateVariables[3]
    app.whiteRookKingSideMoved = stateVariables[4]
    app.whiteRookQueenSideMoved = stateVariables[5]
    app.whiteCastled = stateVariables[6]
    app.blackCastled = stateVariables[7]
    app.lastPiecePlayedOldCoordinate = stateVariables[8]
    app.lastPiecePlayedNewCoordinate = stateVariables[9]
    app.lastPiecePlayed = stateVariables[10]

def resetBoard(app, oldBoard):
    app.boardDict = oldBoard
    board = app.boardDict
    return board

def isCheckmate(app, side, board):
    if getLegalMoves(app, side, board) == []:
        return True
    else:
        return False

def evaluatePosition(board):
    evaluation = 0
    pieceValues = dict()

    # Setting white piece values
    pieceValues['WhitePawn'] = 1.0
    pieceValues['WhiteKnight'] = 3.0
    pieceValues['WhiteBishop'] = 3.0
    pieceValues['WhiteRook'] = 5.0
    pieceValues['WhiteQueen'] = 9.0
    pieceValues['WhiteKing'] = 900.0

    # Setting black piece values
    pieceValues['BlackPawn'] = -1.0
    pieceValues['BlackKnight'] = -3.0
    pieceValues['BlackBishop'] = -3.0
    pieceValues['BlackRook'] = -5.0
    pieceValues['BlackQueen'] = -9.0
    pieceValues['BlackKing'] = -900.0

    # Summing all the piece values together for an evaluation
    for coordinate in board:
        piece = board[coordinate]
        if piece != None:
            key = piece.getSide() + piece.getName()
            evaluation += pieceValues[key]
            
    return evaluation

def minimax(app, board, depth, side, move):
    # Base Case:
    if depth == 0 or isCheckmate(app, side, board):
        #print("Depth is: ", depth)
        evaluation = evaluatePosition(board)
        return evaluation, move

    # Recursive Case
    if side == 'White': # Maximizing Player
        maxEvaluation = -math.inf # negative infinity
        possibleMoves = getLegalMoves(app, side, board)
        
        for move in possibleMoves:
            # Creating copy of old state variables and the board
            oldStateVariables = copy.copy(app.stateVariables)
            oldBoard = copy.copy(board)

            # Updating the board and state variables
            board = updateBoard(move, board)
            updateStateVariables(app, move)
            evaluation, bestMoveFromNode = minimax(app, board, depth-1, 'Black', move)

            # Setting best move and undoing updates to board and state variables
            if evaluation >= maxEvaluation:
                maxEvaluation = evaluation
                bestMove = move
                board = resetBoard(app, oldBoard)
                resetStateVariables(app, oldStateVariables)
            else:
                board = resetBoard(app, oldBoard)
                resetStateVariables(app, oldStateVariables)
        return maxEvaluation, bestMove

    else:
        minEvaluation = math.inf # positive infinity
        possibleMoves = getLegalMoves(app, side, board)

        for move in possibleMoves:
            # Creating copy of old state variables and the board
            oldStateVariables = copy.copy(app.stateVariables)
            oldBoard = copy.copy(board)

            # Updating the board and state variables
            board = updateBoard(move, board)
            updateStateVariables(app, move)
            evaluation, bestMoveFromNode = minimax(app, board, depth - 1, 'White', move)

            # Setting best move and undoing updates to board and state variables
            if evaluation <= minEvaluation:
                minEvaluation = evaluation
                bestMove = move
                board = resetBoard(app, oldBoard)
                resetStateVariables(app, oldStateVariables)
            else:
                board = resetBoard(app, oldBoard)
                resetStateVariables(app, oldStateVariables)
        return minEvaluation, bestMove

def getBestComputerMove(app, givenDepth):
    boardCopy = copy.copy(app.boardDict)
    board = app.boardDict
    side = app.sideToMove
    depth = givenDepth
    move = []
    
    positionEvaluation, bestMove = minimax(app, board, depth, side, move)
    app.boardDict = boardCopy
    return bestMove

def updateBoardFromComputerMove(app, move):
    board = app.boardDict
    for update in move:
        (side, pieceName, coordinate, show) = update
        if show == False:
            board[coordinate] = None
        elif show == True:
            newPiece = ChessPiece(side, pieceName)
            inializeChessPieceGraphics(app, newPiece)
            board[coordinate] = newPiece
    return
