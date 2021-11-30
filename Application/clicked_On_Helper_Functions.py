
#################################################
    # Clicked On Functions
#################################################

def multiplayerClickedOn(app, x, y):
    centerMultiplayerX = app.centerMenuX1
    centerMultiplayerY = app.centerMenuY1

    if (x >= centerMultiplayerX - 100 and 
        x <= centerMultiplayerX + 100 and 
        y >= centerMultiplayerY - 50 and
        y <= centerMultiplayerY + 50):
        return True
    else: return False

def yuviAIClickedOn(app, x, y): 

    centerChessAIX = app.centerMenuX2 
    centerChessAIY = app.centerMenuY2

    if (x >= centerChessAIX - 100 and 
        x <= centerChessAIX + 100 and 
        y >= centerChessAIY - 50 and
        y <= centerChessAIY + 50):
        return True
    else: return False

def loadEvaluationEngineClickedOn(app, x, y):

    centerEvaluationEngineX = app.centerBackgroundX1
    centerEvaluationEngineY = app.centerBackgroundY2

    if (x >= centerEvaluationEngineX - 200 and 
        x <= centerEvaluationEngineX + 200 and 
        y >= centerEvaluationEngineY - 20 and
        y <= centerEvaluationEngineY + 20):
        return True
    else: return False
