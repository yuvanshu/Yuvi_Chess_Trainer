from castling_Helper_Functions import *

#################################################
    # Creating Notation Functions
#################################################
def createNotationDict():
    result = dict()
    result['Knight'] = 'N'
    result['Bishop'] = 'B'
    result['Rook'] = 'R'
    result['Queen'] = 'Q'
    result['King'] = 'K'
    return result

def buildNotation(app, pieceOldSquare, pieceNewSquare, oldCoordinate, newCoordinate):
    notationDict = createNotationDict()
    notation = ''

    # Checking if move made is castling
    if castlingMoveMade(app, pieceOldSquare, newCoordinate, oldCoordinate):
        castlingDirection = getCastlingDirection(newCoordinate)
        if castlingDirection == 'kingside':
            notation = '0-0'
        else:
            notation = '0-0-0'
        return notation
    else:
        # Pieces moving to empty squares
        if pieceNewSquare == None:
            name = pieceOldSquare.getName()
            if name != 'Pawn':
                notation = notationDict[name] + newCoordinate
            else:
                notation = newCoordinate
        else:
            name = pieceOldSquare.getName()
            if name != 'Pawn':
                notation = notationDict[name] + 'x' + newCoordinate
            else:
                notation = oldCoordinate[0] + 'x' + newCoordinate
    if app.isInCheck == True:
        notation += '+'
    return notation
