#################################################
# Setting up the board functions
#################################################

def modelToView(app, row, col):
    squareSize = app.squareSize
    x0 = app.margin + col * squareSize
    y0 = row * squareSize
    x1 = app.margin + (col+1) * squareSize
    y1 = (row+1) * squareSize
    return x0, x1, y0, y1

#Assumes that x and y are within the scope of the board
def viewToModel(app, x, y):
    squareSize = app.squareSize
    row = y // squareSize
    col  = (x - app.margin) // squareSize
    return row, col

def modelToCoordinate(app, row, col):
    orientation = app.boardOrientation
    coordinate = ''
    if orientation == True:
        coordinate = chr(ord('a') + col) + str(app.boardRows-row)
    else:
        coordinate = chr(ord('a') + app.boardCols - col - 1) + str(row + 1)
    return coordinate

def coordinateToModel(app, coordinate):
    orientation = app.boardOrientation
    if orientation == True:
        row = app.boardRows - int(coordinate[1])
        col = ord(coordinate[0]) - ord('a')
    else:
        row = int(coordinate[1]) - 1
        col = app.boardCols - (ord(coordinate[0]) - ord('a') + 1)
    return row, col

def coordinateToLocation(app, coordinate):
    if coordinate != '':
        row, col = coordinateToModel(app, coordinate)
        x0, x1, y0, y1 = modelToView(app, row, col)
        locationX = (x0 + x1) // 2
        locationY = (y0 + y1) // 2
        return locationX, locationY
    else:
        return None

def getSquareColor(app, row, col):
    if (row%2 == col%2):
        return app.boardLightColor
    else:
        return app.boardDarkColor

def highlightSquare(app, coordinate):
    app.highlightSquare = True
    app.highlightSquareCoordinate = coordinate

def unHighlightSquare(app):
    app.highlightSquare = False
    app.highlightSquareCoordinate = None