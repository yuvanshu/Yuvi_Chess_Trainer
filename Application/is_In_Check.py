from piece_Constraint_Helper_Functions import *

#################################################
# Checks if King is In Check
#################################################

#Helper Function:

def getMoves(app, piece):
    side = piece.getSide()
    name = piece.getName()

    if name == 'Pawn':
        return getPawnMoves(app, piece, side)
    elif name == 'Knight':
        return getKnightMoves(app, piece, side)
    elif name == 'Bishop':
        return getBishopMoves(app, piece, side)
    elif name == 'Rook':
        return getRookMoves(app, piece, side)
    elif name == 'Queen':
        return getQueenMoves(app, piece, side)
    elif name == 'King':
        return getKingMoves(app, piece, side)

def isInCheck(app, board, side):

    # Gets coordinate of the king of the side we're concerned with
    for coordinate in board:
        piece = board[coordinate]
        if piece != None and piece.getSide() == side and piece.getName() == 'King':
             kingCoordinate = coordinate
             break

    # Gets moves for each piece (assuming piece is not None)
    # Converts moves to coordinates
    # Checks if king coordinate is in coordinates
    for coordinate in board:
        piece = board[coordinate]
        if piece != None:
            moves = getMoves(app, piece)
            coordinates = getCoordinateList(app, moves)
            if kingCoordinate in coordinates:
                return True
    return False