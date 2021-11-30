#################################################
# Mouse Pressed Helper Functions
#################################################

def viewOnBoard(app, x, y):
    if (x > app.margin and x < app.boardCols * app.squareSize + app.margin
        and y > 0 and y < app.boardRows * app.squareSize):
        return True
    else: return False

def coordinateToPiece(app, coordinate):
    return app.boardDict[coordinate]

def pieceToCoordinate(app, piece):
    for coordinate in app.boardDict:
        if app.boardDict[coordinate] == piece:
            return coordinate
    return None

def pieceToCoordinateGivenBoard(piece, board):
    for coordinate in board:
        if board[coordinate] == piece:
            return coordinate
    return None
