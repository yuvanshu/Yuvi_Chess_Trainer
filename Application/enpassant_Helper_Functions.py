from board_Helper_Functions import *
from mouse_Pressed_Helper_Functions import *

#################################################
# En Passant Helper Functions
#################################################

def getEnPassantSquare(app):

    enPassantCoordinate = ''
    lastPieceNewCoordinate = app.lastPiecePlayedNewCoordinate
    lastPiece = app.lastPiecePlayed

    if lastPiece.getSide() == 'White':
        enPassantCoordinate = lastPieceNewCoordinate[0] + str(int(lastPieceNewCoordinate[1]) - 1)
    else:
        enPassantCoordinate = lastPieceNewCoordinate[0] + str(int(lastPieceNewCoordinate[1]) + 1)
    
    row, col = coordinateToModel(app, enPassantCoordinate)
    return (row, col)

def enPassantMoveMade(app, piece, oldCoordinate, newCoordinate): 

    board = app.boardDict

    # Checks if piece played is a pawn
    if piece.getName() != 'Pawn':
        return False
    
    # Checks if new coordinate of pawn is empty
    if board[newCoordinate] != None:
        return False

    # Checks if new coordinate of pawn is diagonal of old coordinate
    newCoordinateLetter = [chr(ord(oldCoordinate[0]) + 1), chr(ord(oldCoordinate[0]) - 1)]
    if piece.getSide() == 'White':
        newCoordinateNumber = str(int(oldCoordinate[1]) + 1)
        potentialNewCoordinates = [newCoordinateLetter[0] + newCoordinateNumber, newCoordinateLetter[1] + newCoordinateNumber]
        if newCoordinate not in potentialNewCoordinates:
            return False
    else:
        newCoordinateNumber = str(int(oldCoordinate[1]) - 1)
        potentialNewCoordinates = [newCoordinateLetter[0] + newCoordinateNumber, newCoordinateLetter[1] + newCoordinateNumber]
        if newCoordinate not in potentialNewCoordinates:
                return False
    return True

#################################################
# Checks if En Passant Is Possible
#################################################

def canEnPassant(app, piece):

    # Ensures that the piece to perform en passant is a pawn
    if piece.getName() != 'Pawn':
        return False

    lastPiece = app.lastPiecePlayed
    # Ensures that last piece played is not None
    if lastPiece == None:
        return False

    side = piece.getSide()
    pawnCoordinate = pieceToCoordinate(app, piece)

    lastPieceOldCoordinate = app.lastPiecePlayedOldCoordinate
    lastPieceNewCoordinate = app.lastPiecePlayedNewCoordinate

    startingSquaresWhite = {'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2'}
    startingSquaresBlack = {'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7'}

    # Checks if the last move was by a pawn from a starting square 
    if lastPiece.getSide() == 'White':
        if lastPiece.getName() != 'Pawn' or lastPieceOldCoordinate not in startingSquaresWhite:
            return False
    else:
        if lastPiece.getName() != 'Pawn' or lastPieceOldCoordinate not in startingSquaresBlack:
            return False

    # Checks if the pawn to perform en passant is on the right flank
    if side == 'White':
        if pawnCoordinate[1] != '5':
            return False
    else:
        if pawnCoordinate[1] != '4':
            return False

    # Checks if the last pawn that moved is adjacent to pawn to do en passant
    rowLastPiece, colLastPiece = coordinateToModel(app, lastPieceNewCoordinate)
    row, col = coordinateToModel(app, pawnCoordinate)
    if ( not (rowLastPiece == row and colLastPiece == col + 1) and 
        not (rowLastPiece == row and colLastPiece == col - 1)):
        return False

    return True