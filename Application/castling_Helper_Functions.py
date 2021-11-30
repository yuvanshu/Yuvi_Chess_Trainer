from board_Helper_Functions import *
from legal_Moves import *
from is_In_Check import *
#################################################
# Regular Castling Helper Functions
#################################################

# Helper function:

def coordinatesEmpty(app, coordinateList):
    for coordinate in coordinateList:
        piece = coordinateToPiece(app, coordinate)
        if piece != None:
            return False
    return True

def getKingCoordinate(app, side):
    board = app.boardDict
    for coordinate in board:
        piece = board[coordinate]
        if (piece != None and piece.getName() == 'King' and
            piece.getSide() == side):
            return coordinate

def castlingMoveMade(app, piece, newCoordinate, oldCoordinate):
    if piece.getName() != 'King':
        return False
    if ((oldCoordinate == 'e1' and newCoordinate == 'g1') or 
        (oldCoordinate == 'e1' and newCoordinate == 'c1') or
        (oldCoordinate == 'e8' and newCoordinate == 'g8') or
        (oldCoordinate == 'e8' and newCoordinate == 'c8')):
        return True

    # Assuming the piece is a king
    kingCoordinate = getKingCoordinate(app, piece.getSide())
    if kingCoordinate == 'e1' and newCoordinate == 'g1':
        return True
    elif kingCoordinate == 'e1' and newCoordinate == 'c1':
        return True
    elif kingCoordinate == 'e8' and newCoordinate == 'g8':
        return True
    elif kingCoordinate == 'e8' and newCoordinate == 'c8':
        return True
    return False

def getCastlingDirection(coordinate):
    # Assumes that coordinates being passed in are possible castling squares
    if coordinate == 'g1' or coordinate == 'g8':
        return 'kingside'
    else:
        return 'queenside'

#################################################
# Update Helper Functions for Castling
#################################################

def updateKingMoved(app, side):
    if side == 'White':
        app.whiteKingMoved = True
    else:
        app.blackKingMoved = True

def updateRookMoved(app, coordinate):
    # Assumes that rook was on starting square
    if coordinate == 'h1':
        app.whiteRookKingSideMoved = True
    elif coordinate == 'a1':
        app.whiteRookQueenSideMoved = True
    elif coordinate == 'h8':
        app.blackRookKingSideMoved = True
    else:
        app.blackRookQueenSideMoved = True

def updateRookCastled(app, side, castlingDirection):
    if side == 'White' and castlingDirection == 'kingside':
        app.whiteRookKingSideMoved = True
    elif side == 'White' and castlingDirection == 'queenside':
        app.whiteRookQueenSideMoved = True
    elif side == 'Black' and castlingDirection == 'kingside':
        app.blackRookKingSideMoved = True
    else:
        app.blackRookQueenSideMoved = True

def updateSideCastled(app, side):
    if side == 'White':
        app.whiteCastled = False
    else: app.blackCastled = False

def updateRookPosition(app, side, castlingDirection):
    board = app.boardDict
    if side == 'White' and castlingDirection == 'kingside':
        piece = board['h1']
        board['h1'] = None
        board['f1'] = piece
    elif side == 'White' and castlingDirection == 'queenside':
        piece = board['a1']
        board['a1'] = None
        board['d1'] = piece
    elif side == 'Black' and castlingDirection == 'kingside':
        piece = board['h8']
        board['h8'] = None
        board['f8'] = piece
    else:
        piece = board['a8']
        board['a8'] = None
        board['d8'] = piece

#################################################
# Checks if castling is possible
#################################################

def canCastle(app, side, castlingDirection):
    board = app.boardDict
    kingCoordinate = getKingCoordinate(app, side)
    piece = board[kingCoordinate]

    # Checks whether king has moved, is on the starting square, or is in check
    if side == 'White':
        if app.whiteKingMoved == True or kingCoordinate != 'e1' or isInCheck(app, board, side):
            return False
    else:
        if app.blackKingMoved == True or kingCoordinate != 'e8' or isInCheck(app, board, side):
            return False
    
    # Checks whether king can castle given the castling direction

    # Checking if rook has moved or squares between rook and king are empty
    if side == 'White' and castlingDirection == 'kingside':
        coordinatesBetween = ['f1','g1']
        if (app.whiteRookKingSideMoved == True or not coordinatesEmpty(app, coordinatesBetween)): 
            return False
    elif side == 'White' and castlingDirection == 'queenside':
        coordinatesBetween = ['d1','c1']
        if (app.whiteRookQueenSideMoved == True or not coordinatesEmpty(app, coordinatesBetween)): 
            return False 
    elif side == 'Black' and castlingDirection == 'kingside':
        coordinatesBetween = ['f8','g8']
        if (app.blackRookKingSideMoved == True or not coordinatesEmpty(app, coordinatesBetween)): 
            return False
    elif side == 'Black' and castlingDirection == 'queenside':
        coordinatesBetween = ['d8','c8']
        if (app.blackRookQueenSideMoved == True or not coordinatesEmpty(app, coordinatesBetween)): 
            return False

    for coordinate in coordinatesBetween:
        row, col = coordinateToModel(app, coordinate)
        move = (row, col)
        if legalMoveGivenChecks(app, piece, move, board) == False:
            return False
    return True
