#################################################
# Chess Piece Class 
#################################################
class ChessPiece(object):
    def __init__(self, side, piece):
        assert(side == 'Black' or side == 'White')
        pieceSet = {'Knight', 'Bishop', 'King', 'Pawn', 'Queen', 'Rook'}
        assert(piece in pieceSet)
        self.name = piece
        self.side = side

        self.image = None

        self.selected = False
        self.placed = True
    
    def loadImage(self, app):
        imageName = self.side + self.name + '.png'
        self.image = app.loadImage(imageName)

    def scaleImage(self, app, squareSize, scaleConstant):
        if self.image != None:
            pieceSizeDim = max(self.image.height, self.image.width)
            pieceSquareSize = scaleConstant * squareSize
            scalingFactor = (pieceSquareSize / pieceSizeDim)
            self.image = app.scaleImage(self.image, scalingFactor)

    def place(self):
        self.selected = False
        self.placed = True

    def select(self):
        self.selected = True
        self.placed = False

    def getSide(self):
        return self.side

    def getName(self):
        return self.name
