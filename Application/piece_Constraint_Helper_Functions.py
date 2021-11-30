from board_Helper_Functions import *
from mouse_Pressed_Helper_Functions import *

#################################################
# Chess Rule/Constraint functions
#################################################

#Helper Functions
def removeOutOfBoundRowsCols(app, L):
    i = 0
    while (i < len(L)):
        if (L[i][0] < 0 or L[i][0] >= app.boardRows
            or L[i][1] < 0 or L[i][1] >= app.boardCols):
            L.remove(L[i])
        else:
            i += 1
    return L

def isOutOfBounds(app, row, col):
    if (row < 0 or row >= app.boardRows or col < 0 or col >= app.boardCols):
        return True
    else:
        return False

def getCoordinateList(app, L):
    coordinateList = []
    for (row, col) in L:
        coordinate = modelToCoordinate(app, row, col)
        coordinateList.append(coordinate)
    return coordinateList

#################################################
# Get Pawn Moves Function
#################################################

def getPawnMoves(app, piece, side):
    orientation = app.boardOrientation
    if orientation == True:
        factor = 1
    else: factor = -1

    coordinate = pieceToCoordinate(app, piece)
    r, c = coordinateToModel(app, coordinate)

    #Generates moves for pawns based off whether they are on starting squares
    startingSquaresWhite = {'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2'}
    startingSquaresBlack = {'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7'}
    allMoves = []
    if (side == 'White' and coordinate in startingSquaresWhite): 
        allMoves = [(r-1*factor,c), (r-2*factor,c)]
    elif (side == 'Black' and coordinate in startingSquaresBlack):
        allMoves = [(r+1*factor,c), (r+2*factor,c)]
    elif side == 'White':
        allMoves = [(r-1*factor,c)]
    else:
        allMoves = [(r+1*factor,c)]

    #Removing any moves that are out of bounds:
    allMoves = removeOutOfBoundRowsCols(app, allMoves)
    
    # If pawn has promoted, it has no more legal moves
    # allMoves should be empty
    if len(allMoves) == 0:
        return allMoves

    # Removing illegal move squares (piece exists on such square)
    legalMoveSquares = []
    if len(allMoves) == 1:
        (row, col) = allMoves[0]
        square = modelToCoordinate(app, row, col)
        newPiece = coordinateToPiece(app, square)
        if newPiece == None:
            legalMoveSquares.append((row, col))
    else:
        (rowOne, colOne) = allMoves[0]
        (rowTwo, colTwo) = allMoves[1]
        squareOne = modelToCoordinate(app, rowOne, colOne)
        squareTwo = modelToCoordinate(app, rowTwo, colTwo)
        newPieceOne = coordinateToPiece(app, squareOne)
        newPieceTwo = coordinateToPiece(app, squareTwo)
        if newPieceOne == None and newPieceTwo != None:
            legalMoveSquares.append((rowOne, colOne))
        elif newPieceOne == None and newPieceTwo == None:
            legalMoveSquares += allMoves

    # Generating all capture squares for pawns
    if side == 'White':
        captureSquares = [(r-1*factor,c+1), (r-1*factor,c-1)]
    else:
        captureSquares = [(r+1*factor,c+1), (r+1*factor,c-1)]

    #Removing any moves that are out of bounds:
    captureSquares = removeOutOfBoundRowsCols(app, captureSquares)

    # Removing illegal capture squares (no piece or same side piece)
    legalCaptureSquares = []
    for (row, col) in captureSquares:
        square = modelToCoordinate(app, row, col)
        piece = coordinateToPiece(app, square)
        if piece != None and piece.getSide() != side:
            legalCaptureSquares.append((row, col))

    # Combining legal moves squares 
    legalMoves = legalMoveSquares + legalCaptureSquares
    return legalMoves

#################################################
# Get Knight Moves Function
#################################################

def getKnightMoves(app, piece, side):
    coordinate = pieceToCoordinate(app, piece)
    r, c = coordinateToModel(app, coordinate)

    # All possible knight move squares
    allMoves = [(r-2,c+1), (r-2,c-1), (r-1,c+2), (r-1,c-2),
                   (r+2,c+1), (r+2,c-1), (r+1,c+2), (r+1,c-2)]

    # Ensuring moves are all in bounds
    allMoves = removeOutOfBoundRowsCols(app, allMoves)

    # Removing illegal move squares (same side piece on square)
    legalMoveSquares = []
    for move in allMoves:
        (row, col) = move
        square = modelToCoordinate(app, row, col)
        newPiece = coordinateToPiece(app, square)
        if newPiece == None or newPiece.getSide() != side:
            legalMoveSquares.append((row, col))

    return legalMoveSquares

#################################################
# Get Bishop Moves Function
#################################################

def getBishopMoves(app, piece, side):
    coordinate = pieceToCoordinate(app, piece)
    r, c = coordinateToModel(app, coordinate)

    legalMoves = []
    drow, dcol = 1, 1

    #Generating diagonal squares for top rows and right columns
    for i in range(r):
        rowCol = (r - (i+1) * drow, c + (i+1) * dcol)
        (row, col) = rowCol
        if isOutOfBounds(app, row, col) == False:
            legalMoves.append(rowCol)
            square = modelToCoordinate(app, row, col)
            newPiece = coordinateToPiece(app, square)

            if newPiece != None and newPiece.getSide() == side:
                legalMoves.pop()
                break
            elif newPiece != None and newPiece.getSide() != side:
                break

    #Generating diagonal squares for top rows and left columns
    for i in range(r):
        rowCol = (r - (i+1) * drow, c - (i+1) * dcol)
        (row, col) = rowCol
        if isOutOfBounds(app, row, col) == False:
            legalMoves.append(rowCol)
            square = modelToCoordinate(app, row, col)
            newPiece = coordinateToPiece(app, square)

            if newPiece != None and newPiece.getSide() == side:
                legalMoves.pop()
                break
            elif newPiece != None and newPiece.getSide() != side:
                break

    #Generating diagonal squares for bottom rows and right columns
    for i in range(app.boardRows - 1 - r):
        rowCol = (r + (i+1) * drow, c + (i+1) * dcol)
        (row, col) = rowCol
        if isOutOfBounds(app, row, col) == False:
            legalMoves.append(rowCol)
            square = modelToCoordinate(app, row, col)
            newPiece = coordinateToPiece(app, square)

            if newPiece != None and newPiece.getSide() == side:
                legalMoves.pop()
                break
            elif newPiece != None and newPiece.getSide() != side:
                break

    #Generating diagonal squares for bottom rows and left columns
    for i in range(app.boardRows - 1 - r):
        rowCol = (r + (i+1) * drow, c - (i+1) * dcol)
        (row, col) = rowCol
        if isOutOfBounds(app, row, col) == False:
            legalMoves.append(rowCol)
            square = modelToCoordinate(app, row, col)
            newPiece = coordinateToPiece(app, square)

            if newPiece != None and newPiece.getSide() == side:
                legalMoves.pop()
                break
            elif newPiece != None and newPiece.getSide() != side:
                break
        
    return legalMoves

#################################################
# Get Rook Moves Function
#################################################

def getRookMoves(app, piece, side):
    coordinate = pieceToCoordinate(app, piece)
    r, c = coordinateToModel(app, coordinate)

    legalMoves = []
    drow, dcol = 1, 1

    #Generating squares on the file on the top
    for i in range(r):
        rowCol = (r - (i+1) * drow, c + (0) * dcol)
        (row, col) = rowCol

        if isOutOfBounds(app, row, col) == False:
            legalMoves.append(rowCol)
            square = modelToCoordinate(app, row, col)
            newPiece = coordinateToPiece(app, square)

            if newPiece != None and newPiece.getSide() == side:
                legalMoves.pop()
                break
            elif newPiece != None and newPiece.getSide() != side:
                break

    #Generating squares on the file on the bottom          
    for i in range(app.boardRows - 1 - r):
        rowCol = (r + (i+1) * drow, c + (0) * dcol)
        (row, col) = rowCol

        if isOutOfBounds(app, row, col) == False:
            legalMoves.append(rowCol)
            square = modelToCoordinate(app, row, col)
            newPiece = coordinateToPiece(app, square)

            if newPiece != None and newPiece.getSide() == side:
                legalMoves.pop()
                break
            elif newPiece != None and newPiece.getSide() != side:
                break

    #Generating squares in the row to the left
    for i in range(c):
        rowCol = (r + (0) * drow, c - (i+1) * dcol)
        (row, col) = rowCol

        if isOutOfBounds(app, row, col) == False:
            legalMoves.append(rowCol)
            square = modelToCoordinate(app, row, col)
            newPiece = coordinateToPiece(app, square)

            if newPiece != None and newPiece.getSide() == side:
                legalMoves.pop()
                break
            elif newPiece != None and newPiece.getSide() != side:
                break

    #Generating squares in the row to the right
    for i in range(app.boardCols - 1 - c):
        rowCol = (r + (0) * drow, c + (i+1) * dcol)
        (row, col) = rowCol

        if isOutOfBounds(app, row, col) == False:
            legalMoves.append(rowCol)
            square = modelToCoordinate(app, row, col)
            newPiece = coordinateToPiece(app, square)

            if newPiece != None and newPiece.getSide() == side:
                legalMoves.pop()
                break
            elif newPiece != None and newPiece.getSide() != side:
                break
        
    return legalMoves


#################################################
# Get Queen Moves Function
#################################################

def getQueenMoves(app, piece, side):
    legalMoves = []
    #Add all diagonal squares
    legalMoves += getBishopMoves(app, piece, side)
    #Add all file and row squares
    legalMoves += getRookMoves(app, piece, side)
    return legalMoves

#################################################
# Get King Moves Function
#################################################

def getKingMoves(app, piece, side):
    coordinate = pieceToCoordinate(app, piece)
    r, c = coordinateToModel(app, coordinate)

    # Generate all possible squares for king to move to
    allMoves = []
    directions =  [-1,0,1]
    for row in range(len(directions)):
        for col in range(len(directions)):
            allMoves.append((r + directions[row], c + directions[col]))
    
    # Remove squares that are out of bounds
    allMoves = removeOutOfBoundRowsCols(app, allMoves)
    
    # Find legal moves (square is empty or contains piece of opposite side)
    legalMoves = []
    for move in allMoves:
        (row, col) = move
        square = modelToCoordinate(app, row, col)
        newPiece = coordinateToPiece(app, square)
        if newPiece == None or newPiece.getSide() != side:
            legalMoves.append((row, col))

    return legalMoves