from board_Helper_Functions import *
from piece_Constraint_Helper_Functions import *
from is_In_Check import *
from enpassant_Helper_Functions import *
from castling_Helper_Functions import *

#################################################
# Checks if move made is legal given possible checks
#################################################

def legalMoveGivenChecks(app, piece, move, board):

    # Determine new coordinate piece is moved to
    (row, col) = move
    newCoordinate = modelToCoordinate(app, row, col)

    # Determine old coordinate of the piece
    for coordinate in board:
        if board[coordinate] == piece:
            oldCoordinate = coordinate
            break
    # Determine the piece that occupies new coordinate (can be None)
    pieceNewCoordinate = board[newCoordinate]
    
    #Update the board given the move
    board[newCoordinate] = piece
    board[oldCoordinate] = None

    # Check if side in updated board is in check
    # Return False if it is; True if it isn't
    if isInCheck(app, board, piece.getSide()):
        board[newCoordinate] = pieceNewCoordinate
        board[oldCoordinate] = piece
        return False
    else:
        board[newCoordinate] = pieceNewCoordinate
        board[oldCoordinate] = piece
        return True

#########################################################
# Returns all legal moves for piece given possible checks
#########################################################

# Helper functions

# Accepts coordinate and a list containing coordinates
def coordinateInList(coordinate, L):
    if coordinate in L:
        return True
    else:
        return False

def legalMovesGivenChecks(app, piece, moveList, board):
    legalMoves = []
    for move in moveList:
        if legalMoveGivenChecks(app, piece, move, board):
            legalMoves.append(move)
    return legalMoves

def checkLegalMove(app, piece, coordinate, oldCoordinate):

    # Checking if move is played by side to move
    if piece.getSide() != app.sideToMove:
        return False

    # Create board copy
    board = app.boardDict
    
    # If piece played is a pawn:
    if piece.getName() == 'Pawn':
        pawnMoves = getPawnMoves(app, piece, piece.getSide())
        pawnMoves = legalMovesGivenChecks(app, piece, pawnMoves, board)

        #Checks if en passant is possible and adds it to pawnMoves if it is
        if canEnPassant(app, piece):
            pawnMoves.append(getEnPassantSquare(app))

        pawnLegalCoordinates = getCoordinateList(app, pawnMoves)
        return coordinateInList(coordinate, pawnLegalCoordinates)
    
    # If piece played is a knight:
    elif piece.getName() == 'Knight':
        knightMoves = getKnightMoves(app, piece, piece.getSide())
        knightMoves = legalMovesGivenChecks(app, piece, knightMoves, board)
        knightLegalCoordinates = getCoordinateList(app, knightMoves)
        return coordinateInList(coordinate, knightLegalCoordinates)

    # If piece played is a bishop:
    elif piece.getName() == 'Bishop':
        bishopMoves = getBishopMoves(app, piece, piece.getSide())
        bishopMoves = legalMovesGivenChecks(app, piece, bishopMoves, board)
        bishopLegalCoordinates = getCoordinateList(app, bishopMoves)
        return coordinateInList(coordinate, bishopLegalCoordinates)

    # If piece played is a rook:
    elif piece.getName() == 'Rook':
        rookMoves = getRookMoves(app, piece, piece.getSide())
        rookMoves = legalMovesGivenChecks(app, piece, rookMoves, board)
        rookLegalCoordinates = getCoordinateList(app, rookMoves)
        return coordinateInList(coordinate, rookLegalCoordinates)

    # If piece played is a queen:
    elif piece.getName() == 'Queen':
        queenMoves = getQueenMoves(app, piece, piece.getSide())
        queenMoves = legalMovesGivenChecks(app, piece, queenMoves, board)
        queenLegalCoordinates = getCoordinateList(app, queenMoves)
        return coordinateInList(coordinate, queenLegalCoordinates)

    # If piece played is a king:
    elif piece.getName() == 'King':
        kingMoves = getKingMoves(app, piece, piece.getSide())
        kingMoves = legalMovesGivenChecks(app, piece, kingMoves, board)

        # Checks if move is castling and adds it to moves if legal
        if castlingMoveMade(app, piece, coordinate, oldCoordinate):
            castlingDirection = getCastlingDirection(coordinate)
            if canCastle(app, piece.getSide(), castlingDirection):
                row, col = coordinateToModel(app, coordinate)
                move = (row, col)
                kingMoves.append(move)

        kingLegalCoordinates = getCoordinateList(app, kingMoves)
        return coordinateInList(coordinate, kingLegalCoordinates)