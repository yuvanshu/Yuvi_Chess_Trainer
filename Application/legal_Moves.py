from board_Helper_Functions import *
from is_In_Check import *
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