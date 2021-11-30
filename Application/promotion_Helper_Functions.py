from chessPieceClass import *
from inializer_Helper_Functions import *

#################################################
# Promotion Helper Functions
#################################################

def getPromotionPiece(app, side):

    # Retrieves promotion piece as input from user
    promotionPiece = app.getUserInput("Select piece for promotion: Choose from 'Knight', 'Bishop', 'Rook', or 'Queen'")

    # If promotion piece input is not valid, ask user again until it is
    pieceSet = {'Knight', 'Bishop', 'King', 'Pawn', 'Queen', 'Rook'}
    while (promotionPiece not in pieceSet):
        promotionPiece = app.getUserInput("Please select valid input: Choose from 'Knight', 'Bishop', 'Rook', or 'Queen'")

    # Create new chess piece object from promotion piece input and set it to app.promotionPiece
    app.pieceToBePromoted = ChessPiece(side, promotionPiece)
    inializeChessPieceGraphics(app, app.pieceToBePromoted)

def placePromotionPiece(app, coordinate):
    board = app.boardDict
    board[coordinate] = app.pieceToBePromoted

#################################################
# Checks if Promotion Occurred
#################################################

def piecePromoted(piece, coordinate):
    # Ensures that the piece is a pawn
    if piece == None or piece.getName() != 'Pawn':
        return False 

    side = piece.getSide()

    # Checks if pawn is on eigth or first rank depending on side
    if side == 'White':
        if coordinate[1] != '8':
            return False
    else:
        if coordinate[1] != '1':
            return False
    return True