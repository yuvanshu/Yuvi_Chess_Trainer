from chessPieceClass import *

def inializeBoardDict(app):
    app.boardDict = dict()

    letterSet = {'a','b','c','d','e','f','g','h'}
    numberSet = {'1','2','3','4','5','6','7','8'}

    for letter in letterSet:
        for number in numberSet:
            coordinate = letter + number
            app.boardDict[coordinate] = None
    
    for coordinate in app.boardDict:
        # Inializing White Pieces
        if coordinate in ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2']:
            app.boardDict[coordinate] = ChessPiece('White', 'Pawn')
        elif coordinate in ['a1', 'h1']:
            app.boardDict[coordinate] = ChessPiece('White', 'Rook')
        elif coordinate in ['b1', 'g1']:
            app.boardDict[coordinate] = ChessPiece('White', 'Knight')
        elif coordinate in ['c1', 'f1']:
            app.boardDict[coordinate] = ChessPiece('White', 'Bishop')
        elif coordinate in ['d1']:
            app.boardDict[coordinate] = ChessPiece('White', 'Queen')
        elif coordinate in ['e1']:
            app.boardDict[coordinate] = ChessPiece('White', 'King')

        # Inializing Black Pieces
        if coordinate in ['a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7']:
            app.boardDict[coordinate] = ChessPiece('Black', 'Pawn')
        elif coordinate in ['a8', 'h8']:
            app.boardDict[coordinate] = ChessPiece('Black', 'Rook')
        elif coordinate in ['b8', 'g8']:
            app.boardDict[coordinate] = ChessPiece('Black', 'Knight')
        elif coordinate in ['c8', 'f8']:
            app.boardDict[coordinate] = ChessPiece('Black', 'Bishop')
        elif coordinate in ['d8']:
            app.boardDict[coordinate] = ChessPiece('Black', 'Queen')
        elif coordinate in ['e8']:
            app.boardDict[coordinate] = ChessPiece('Black', 'King')

def inializeAllChessPiecesGraphics(app):
    for coordinate in app.boardDict:
        piece = app.boardDict[coordinate]
        if piece != None:
            inializeChessPieceGraphics(app, piece)

def inializeChessPieceGraphics(app, piece):
    squareSize = app.squareSize
    piece.loadImage(app)
    if piece.getName() == 'Knight' or piece.getName() == 'Queen':
        piece.scaleImage(app, squareSize, 1)
    else:
        piece.scaleImage(app, squareSize, 4/5)