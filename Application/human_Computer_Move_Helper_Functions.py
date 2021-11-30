from board_Helper_Functions import *
from mouse_Pressed_Helper_Functions import *
from legal_Moves_Given_Checks import *
from promotion_Helper_Functions import *
from custom_Chess_Evaluation_Helper_Functions import *
from notation_Helper_Functions import *

#################################################
    # Human-Computer Move Functions
#################################################
def makeHumanMove(app, x, y):
    board = app.boardDict

    # Determines if human is in checkmate 
    if isCheckmate(app, app.sideToMove, board):
        app.isCheckmate = True
        app.computerModeOn = False

    # Allows human user to make moves on board
    # Ensures that mouse clicked on the board
    if viewOnBoard(app, x, y):
        row, col = viewToModel(app, x, y)
        coordinate = modelToCoordinate(app, row, col)

        if app.pieceSelected == False:
            piece = coordinateToPiece(app, coordinate)
            if piece != None:
                highlightSquare(app, coordinate)
                app.pieceToMove = piece
                app.pieceSelected = True
                piece.select()
        else:
            oldCoordinate = pieceToCoordinate(app, app.pieceToMove)
            pieceNewCoordinate = board[coordinate]
            if checkLegalMove(app, app.pieceToMove, coordinate, oldCoordinate) == True:
                #Checks if en passant move made
                if enPassantMoveMade(app, app.pieceToMove, oldCoordinate, coordinate):
                    app.boardDict[app.lastPiecePlayedNewCoordinate] = None

                piece = coordinateToPiece(app, coordinate)
                if piece != None:
                    app.boardDict[coordinate] = None
                app.boardDict[coordinate] = app.pieceToMove
                app.boardDict[oldCoordinate] = None

                if app.sideToMove == 'White':
                    app.sideToMove = 'Black'
                else: app.sideToMove = 'White'

                app.pieceToMove.place()

                # Performs promotion if piece selected for promotion
                if piecePromoted(app.pieceToMove, coordinate):
                    getPromotionPiece(app, app.pieceToMove.getSide())
                    placePromotionPiece(app, coordinate)
                
                # Checks if piece placed is a rook
                if app.pieceToMove.getName() == 'Rook':
                    updateRookMoved(app, oldCoordinate)

                # Checks if rooks are on board
                rookCoordinates = ['a1', 'h1', 'a8', 'h8']
                for rookCoordinate in rookCoordinates:
                    piece = board[rookCoordinate]
                    if (rookCoordinate == 'a1' or rookCoordinate == 'h1') and (piece == None or piece.getSide() != 'White' or piece.getName() != 'Rook'):
                        updateRookMoved(app, rookCoordinate)
                    elif (rookCoordinate == 'a8' or rookCoordinate == 'h8') and (piece == None or piece.getSide() != 'Black' or piece.getName() != 'Rook'):
                        updateRookMoved(app, rookCoordinate)

                # Checks if piece placed is a king
                if app.pieceToMove.getName() == 'King':
                    # Update that king on specific side has been moved
                    updateKingMoved(app, app.pieceToMove.getSide())

                    # Checks if move played is a castling move
                    # Runs subsequent functions on it
                    if castlingMoveMade(app, app.pieceToMove, coordinate, oldCoordinate):
                        castlingDirection = getCastlingDirection(coordinate)
                        updateSideCastled(app, app.pieceToMove.getSide())
                        updateRookCastled(app, app.pieceToMove.getSide(), castlingDirection)
                        updateRookPosition(app, app.pieceToMove.getSide(), castlingDirection)

                #Updates last piece played and its old and new coordinates
                app.lastPiecePlayed = app.pieceToMove
                app.lastPiecePlayedOldCoordinate = oldCoordinate
                app.lastPiecePlayedNewCoordinate = coordinate

                # Determines if other player is in check
                check = isInCheck(app, board, app.sideToMove)
                if check == True:
                    app.isInCheck = True
                else:
                    app.isInCheck = False

                # Determines if checkmate has occurred after move played
                if isCheckmate(app, app.sideToMove, board):
                    app.isCheckmate = True
                    app.computerModeOn = False

                notation = buildNotation(app, app.pieceToMove, pieceNewCoordinate, oldCoordinate, coordinate)
                app.chessNotation.append(notation)
                #print(app.chessNotation)

                app.pieceToMove = None
                app.pieceSelected = False
                unHighlightSquare(app)

                #Ensures that it is now computer's turn to play and not human's
                if app.computerModeOn == True:
                    app.computerToMove = True
                    app.humanToMove = False
                
                # Build tensor for generating evaluation
                featureTensor = createEvaluationInput(app)

                # Generate Evaluation
                evaluation = generateEvaluation(app, featureTensor)
                app.chessEvaluations.append(evaluation)

            else:
                app.pieceSelected = False
                app.pieceToMove = None
                unHighlightSquare(app)
    else:
        app.pieceSelected = False
        app.pieceToMove = None
        unHighlightSquare(app)

    #Ensure state variables are always current
    ensureCurrentStateVariables(app)

def makeComputerMove(app):
    board = app.boardDict
    depth = 2 # Set depth considering trade off between time and move quality
    bestMove = getBestComputerMove(app, depth)

    # Updating state variables after retrieving move
    piecePlayed = ChessPiece(bestMove[1][0], bestMove[1][1])
    oldCoordinate = bestMove[0][2]
    coordinate = bestMove[1][2]
    pieceNewCoordinate = board[coordinate]

    updateBoardFromComputerMove(app, bestMove)
    
    # Checks if piece placed is a rook
    if piecePlayed.getName() == 'Rook':
        updateRookMoved(app, oldCoordinate)

    # Checks if rooks are on board
    rookCoordinates = ['a1', 'h1', 'a8', 'h8']
    for rookCoordinate in rookCoordinates:
        piece = board[rookCoordinate]
        if (rookCoordinate == 'a1' or rookCoordinate == 'h1') and (piece == None or piece.getSide() != 'White' or piece.getName() != 'Rook'):
            updateRookMoved(app, rookCoordinate)
        elif (rookCoordinate == 'a8' or rookCoordinate == 'h8') and (piece == None or piece.getSide() != 'Black' or piece.getName() != 'Rook'):
            updateRookMoved(app, rookCoordinate)

    # Checks if piece placed is a king
    if piecePlayed.getName() == 'King':
        # Update that king on specific side has been moved
        updateKingMoved(app, piecePlayed.getSide())

    # Checks if move is played is a castling move
    # Runs subsequent functions on it
    if castlingMoveMade(app, piecePlayed, coordinate, oldCoordinate):
        castlingDirection = getCastlingDirection(coordinate)
        updateSideCastled(app, piecePlayed.getSide())
        updateRookCastled(app, piecePlayed.getSide(), castlingDirection)
        updateRookPosition(app, piecePlayed.getSide(), castlingDirection)
    
    app.lastPiecePlayed = piecePlayed
    app.lastPiecePlayedOldCoordinate = oldCoordinate
    app.lastPiecePlayedNewCoordinate = coordinate

    # Ensures that it is now human's turn to play and not computer's
    app.computerToMove = False
    app.humanToMove = True

    # Switching side to move
    if app.sideToMove == 'White':
        app.sideToMove = 'Black'
    else: app.sideToMove = 'White'

    # Determines if other player is in check
    check = app.isInCheck
    if check == True:
        app.isInCheck = False

    notation = buildNotation(app, piecePlayed, pieceNewCoordinate, oldCoordinate, coordinate)
    app.chessNotation.append(notation)

    #Ensure state variables are always current
    ensureCurrentStateVariables(app)

    # Build tensor for generating evaluation
    featureTensor = createEvaluationInput(app)

    # Generate Evaluation
    evaluation = generateEvaluation(app, featureTensor)
    app.chessEvaluations.append(evaluation)

