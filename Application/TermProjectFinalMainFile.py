from cmu_112_graphics import *

from chessPieceClass import *
from ml_Model_Class_Functions import *

from inializer_Helper_Functions import *
from board_Helper_Functions import *
from piece_Constraint_Helper_Functions import *
from castling_Helper_Functions import *
from enpassant_Helper_Functions import *
from promotion_Helper_Functions import *
from is_In_Check import *
from legal_Moves import *
from legal_Moves_Given_Checks import *
from minimax_Helper_Functions import *
from custom_Chess_Evaluation_Helper_Functions import *
from notation_Helper_Functions import *
from mouse_Pressed_Helper_Functions import *
from human_Computer_Move_Helper_Functions import *
from clicked_On_Helper_Functions import *
from draw_Functions import *

import math
import decimal

#################################################
# Citations
#################################################
# Black Bishop Image Citation: https://pngset.com/download-free-png-pazad
# Black Pawn Image Citation: https://favpng.com/png_view/chess-avatar-clip-art-png/vef73JR3
# Black Knight Image Citation: https://thenounproject.com/term/chess-knight/108491/
# Black King Image Citation: https://flyclipart.com/chess-king-piece-clip-art-free-vector-knight-chess-piece-clipart-910242
# Black Rook Image Citation: https://www.pngegg.com/en/png-ksxat
# Black Queen Image Citation: https://www.iconspng.com/image/112202/chess-tile-queen

# Note: White Piece Image Citations not included. This is because black piece images were simply filled with white
# color. No new image sources were used. 

# Background Screen Image Citation: https://wallpaperaccess.com/cool-chess
# Menu Screen Image Citation: https://www.peakpx.com/en/hd-wallpaper-desktop-ouoxb

# cmu_112_graphics Citation: https://www.cs.cmu.edu/~112/index.html
# This graphics file is developed and distributed by Carnegie Mellon Course: 15-112

#################################################
# Inializer (appStarted) function
#################################################

def appStarted(app):
    app.margin = 50 #For providing space for displaying chess cordinates
    app.boardOrientation = True
    app.boardDarkColor = 'navyblue'
    app.boardLightColor = 'gray'
    app.boardRows = 8
    app.boardCols = 8
    app.sideSpace = app.width - app.height #Default will be 250 pixels

    app.squareSize = (app.height - app.margin)//app.boardRows

    app.pieceSelected = False
    app.pieceToMove = None

    app.highlightSquare = False
    app.highlightSquareCoordinate = None

    app.sideToMove = 'White'
    app.isInCheck = False
    app.isCheckmate = False

    #################################################
    # Initializing App Variables for Castling Function
    #################################################
    app.whiteKingMoved = False
    app.blackKingMoved = False

    app.blackRookKingSideMoved = False
    app.blackRookQueenSideMoved = False
    app.whiteRookKingSideMoved = False
    app.whiteRookQueenSideMoved = False

    app.whiteCastled = False
    app.blackCastled = False

    #################################################
    # Initializing App Variables for En Passant Function
    #################################################

    app.lastPiecePlayedOldCoordinate = None
    app.lastPiecePlayedNewCoordinate = None
    app.lastPiecePlayed = None

    #################################################
    # Initializing App Variables for Promotion Function
    #################################################

    app.pieceToBePromoted = None

    #################################################
    # Initializing State Variables List for Min-Max Function
    #################################################
    app.stateVariables = [app.whiteKingMoved, app.blackKingMoved, app.blackRookKingSideMoved,
                        app.blackRookQueenSideMoved, app.whiteRookKingSideMoved, 
                        app.whiteRookQueenSideMoved, app.whiteCastled, app.blackCastled,
                        app.lastPiecePlayedOldCoordinate, app.lastPiecePlayedNewCoordinate,
                        app.lastPiecePlayed]

    #################################################
    # Keeping Track of Computer Move Variables
    #################################################

    app.computerToMove = False
    app.humanToMove = True
    app.computerModeOn = None

    #################################################
    # Keeping track of application screens
    #################################################

    app.backgroundScreen = True
    app.mainMenuScreen = False
    app.multiplayerModeScreen = False
    app.yuviChessAIScreen = False

    #################################################
    # Chess Notation List
    #################################################

    app.chessNotation = []
    app.chessEvaluations = []

    #################################################
    # Loading Chess Evaluation ML Model
    #################################################

    app.filepath = '/Users/yuvanshuagarwal/Desktop/Fundamentals of Programming 15-112/TermProject/model_cpu'
    app.filepath2 = '/Users/yuvanshuagarwal/Desktop/Fundamentals of Programming 15-112/TermProject/model_cpu2'
    app.loaded_model = joblib.load(app.filepath2)

    #################################################
    # Loading Chess Background Image
    #################################################
    
    app.imageName = 'TermProjectChessWallPaper4.png'
    app.wallpaperImage = app.loadImage(app.imageName)
    app.wallpaperImage = app.scaleImage(app.wallpaperImage, 0.8)

    app.centerBackgroundX1 = 575
    app.centerBackgroundY2 = 675

    #################################################
    # Loading Chess Menu Screen Image
    #################################################
    
    app.imageName2 = 'TermProjectChessMenu2.jpeg'
    app.chessMenuImage = app.loadImage(app.imageName2)
    app.chessMenuImage = app.scaleImage(app.chessMenuImage, 0.75)
    
    # Key variables
    app.centerMenuX1 = 475
    app.centerMenuX2 = 775
    app.centerMenuY1 = 175
    app.centerMenuY2 = 175

    #################################################
    # Initializing Chess Board Dictionary
    #################################################
    inializeBoardDict(app)
    inializeAllChessPiecesGraphics(app)

#################################################
# Keyboard/Mouse Event Functions
#################################################

def keyPressed(app, event):
    if (event.key == 'f'): #f for flip
        app.boardOrientation = not app.boardOrientation
    elif (event.key == 'r'): #r for restart
        appStarted(app)

def mousePressed(app, event):
    x = event.x
    y = event.y

    if app.backgroundScreen == True and loadEvaluationEngineClickedOn(app, x, y):
        app.mainMenuScreen = True
        app.backgroundScreen = False
    if app.mainMenuScreen == True and yuviAIClickedOn(app, x, y):
        app.computerModeOn = True
        app.mainMenuScreen = False
        app.yuviChessAIScreen = True
    elif app.mainMenuScreen == True and multiplayerClickedOn(app, x, y):
        app.computerModeOn = False
        app.mainMenuScreen = False
        app.multiplayerModeScreen = True

    if app.multiplayerModeScreen == True or app.yuviChessAIScreen == True:
        if app.humanToMove == True:
            makeHumanMove(app, x, y)

#################################################
# Timer Fired Function
#################################################
def timerFired(app):
    if app.computerModeOn == True and app.computerToMove == True:
        makeComputerMove(app)

#################################################
# RedrawAll Function
#################################################
def redrawAll(app, canvas):
    if app.backgroundScreen == True:
        drawChessBackground(app, canvas)
    elif app.mainMenuScreen == True:
        drawChessMenuScreen(app, canvas)
    elif app.yuviChessAIScreen == True or app.multiplayerModeScreen == True:
        drawBoard(app, canvas)
        drawCoordinates(app, canvas)
        drawHighlightedSquare(app, canvas)
        drawChessPieces(app, canvas)
        drawChessNotation(app, canvas)
        drawChessAnalysis(app, canvas)
        drawSideToMove(app, canvas)
        drawEvaluation(app, canvas)
        drawCheckmate(app, canvas)

runApp(width=1250, height=750)
