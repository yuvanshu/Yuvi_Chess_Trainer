from board_Helper_Functions import *
from cmu_112_graphics import *

#################################################
# Draw Functions:
#################################################
def drawBoard(app, canvas):
    for row in range(app.boardRows):
        for col in range(app.boardCols):
            x0, x1, y0, y1 = modelToView(app, row, col)
            squareColor = getSquareColor(app, row, col)
            canvas.create_rectangle(x0, y0, x1, y1, fill = squareColor)

def drawCoordinates(app, canvas):
    orientation = app.boardOrientation
    squareSize = app.squareSize

    for row in range(app.boardRows):
        y0 = row * squareSize
        y1 = (row+1) * squareSize
        if orientation == True:
            canvas.create_text(app.margin//2, (y0 + y1)//2, 
                               text=f'{app.boardRows-row}', 
                               fill='navyblue', font='Arial 20 bold')
        else:
            canvas.create_text(app.margin//2, (y0 + y1)//2, 
                               text=f'{row + 1}', 
                               fill='navyblue', font='Arial 20 bold')
    for col in range(app.boardCols):
        x0 = app.margin + col * squareSize
        x1 = app.margin + (col+1) * squareSize
        if orientation == True:
            letter = chr(ord('a') + col)
            canvas.create_text((x0 + x1)//2, 
                                app.margin//2 + app.boardRows * squareSize, 
                                text=f'{letter}', 
                               fill='navyblue', font='Arial 20 bold')
        else:
            letter = chr(ord('a') + app.boardCols - col - 1)
            canvas.create_text((x0 + x1)//2, 
                               app.margin//2 + app.boardRows * squareSize, 
                               text=f'{letter}', 
                               fill='navyblue', font='Arial 20 bold')        

def drawChessPieces(app, canvas):
    for coordinate in app.boardDict:
        if app.boardDict[coordinate] != None:
            piece = app.boardDict[coordinate]
            locationX, locationY = coordinateToLocation(app, coordinate)
            canvas.create_image(locationX, locationY, image=ImageTk.PhotoImage(piece.image))

def drawHighlightedSquare(app, canvas):
    coordinate = app.highlightSquareCoordinate
    if app.highlightSquare == True and coordinate != None:
        row, col = coordinateToModel(app, coordinate)
        x0, x1, y0, y1 = modelToView(app, row, col)
        canvas.create_rectangle(x0, y0, x1, y1, fill = 'tomato2')

def drawCheckmate(app, canvas):
    centerX = (app.squareSize * app.boardRows) + app.margin + app.sideSpace // 2
    centerY = 50
    if app.isCheckmate == True:
        canvas.create_text(centerX, centerY, text=f'Checkmate! Press r to restart', fill='navyblue', font='Arial 20 bold')

def drawSideToMove(app, canvas):
    centerX = (app.squareSize * app.boardRows) + app.margin + app.sideSpace // 4
    centerY = 100
    canvas.create_text(centerX, centerY, text=f'{app.sideToMove} to play', fill='navyblue', font='Arial 20 bold')

def drawEvaluation(app, canvas):
    evaluations = app.chessEvaluations
    centerX = (app.squareSize * app.boardRows) + app.margin + (app.sideSpace * (3/4))
    centerY = 100
    if len(evaluations) > 0:
        evaluation = round(evaluations[-1], 2)
        canvas.create_text(centerX - 25, centerY, text='Evaluation: ', fill='navyblue', font='Arial 20 bold')
        canvas.create_text(centerX + 50, centerY, text=evaluation, fill='navyblue', font='Arial 20 bold')

def drawChessNotation(app, canvas):
    canvas.create_rectangle(app.boardRows * app.squareSize + app.margin, 125, app.width, 125, fill = 'navyblue')
    chessNotation = app.chessNotation
    plyCount = 0
    centerX = app.boardRows * app.squareSize + 100
    centerY = 150
    for i in range(len(chessNotation)):
        notation = chessNotation[i]
        if i < len(chessNotation) - 1:
            notationLength = len(chessNotation[i + 1])
        else:
            notationLength = len(chessNotation[i])
        if plyCount % 2 == 0:
            moveNumber = plyCount // 2 + 1
            canvas.create_text(centerX, centerY, text=f'{moveNumber}. {notation}', fill='navyblue', font='Arial 18 bold')
            centerX += 25 + 15 * notationLength - 2
        else:
            canvas.create_text(centerX, centerY, text=f'{notation}', fill='navyblue', font='Arial 18 bold')
            centerX += 25 + 15 * notationLength - 2
        if centerX >= app.width - 45:
            centerX = app.boardRows * app.squareSize + 100 
            centerY += 25
        plyCount += 1

def drawChessAnalysis(app, canvas):
    plyCount = 0
    evaluations = app.chessEvaluations
    chessNotation = app.chessNotation
    messages = []
    centerX = (app.boardRows * app.squareSize + 100 + app.width) // 2
    centerY = 500
    if len(evaluations) >= 2:
        for i in range(len(evaluations)-2):
            index = i + 2
            difference = evaluations[index] - evaluations[index - 2]
            if plyCount % 2 == 0:
                if difference <= -1.0:
                    message = f'Mistake occurred on {plyCount // 2 + 1}. {chessNotation[plyCount]} Evaluation: {evaluations[index]}'
                    messages.append(message)
            else:
                if difference >= 1.0:
                    message = f'Mistake occurred on {plyCount // 2 + 1}... {chessNotation[plyCount]} Evaluation: {evaluations[index]}'
                    messages.append(message)
            plyCount += 1
        if len(messages) > 0:
            for message in messages:
                canvas.create_text(centerX, centerY, text=message, fill='navyblue', font='Arial 18 bold')
                centerY += 25
        else:
            canvas.create_text(centerX, centerY, text='No Mistakes Found', fill='navyblue', font='Arial 18 bold')
    else:
        canvas.create_text(centerX, centerY, text='Plycount of 2 or more needed for analysis', fill='navyblue', font='Arial 18 bold')

def drawChessBackground(app, canvas):
    centerX = 550
    centerY = 550
    canvas.create_image(app.width//2, app.height//2, image=ImageTk.PhotoImage(app.wallpaperImage))
    canvas.create_text(centerX, centerY, text='Yuvi', fill='black', font='Helvetica 120 bold')
    canvas.create_text(centerX + 25, centerY + 75, text='Your Chess Trainer', fill='black', font='Helvetica 40 bold')
    rectCenterX = centerX + 25 
    rectCenterY = centerY + 125
    canvas.create_rectangle(rectCenterX - 200, rectCenterY - 20, rectCenterX + 200, rectCenterY + 20, fill = 'black')
    canvas.create_text(rectCenterX, rectCenterY, text='Click to Start', fill='white', font='Helvetica 25 bold')

def drawChessMenuScreen(app, canvas):
    centerX1 = app.centerMenuX1 
    centerX2 = app.centerMenuX2 
    centerY1 = app.centerMenuY1 
    centerY2 = app.centerMenuY2

    canvas.create_rectangle(0, 0, app.width, app.height, fill = 'black')
    canvas.create_image(app.width//2, app.height//2, image=ImageTk.PhotoImage(app.chessMenuImage))
    canvas.create_rectangle(centerX1 - 100, centerY1 - 50, centerX1 + 100, centerY1 + 50, fill = 'white')
    canvas.create_rectangle(centerX2 - 100, centerY2 - 50, centerX2 + 100, centerY2 + 50, fill = 'black')
    canvas.create_text(centerX1, centerY1, text='Multiplayer', fill='black', font='Helvetica 25 bold')
    canvas.create_text(centerX2, centerY2, text='Play Yuvi', fill='white', font='Helvetica 25 bold')
