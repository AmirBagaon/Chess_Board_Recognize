import time
import cv2
from chessboard_to_fen import get_FEN, generate_pieces
from pynput import keyboard
from PIL import ImageGrab

start_time = time.time()
cropped_board_path = r"<Some path for the cropped board output image>"
screenshot_path = r"<Some path for the screenshot image>"
classifcation_pieces_output_path = r"<Some path for the output of the pieces images>"


def get_board(image_path):
    """
    Locates and crops the board from the screenshot
    """
    print("Getting image")
    # take_screenshot(image_path)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 17)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,7,3)
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Find bounding box and extract ROI
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = image[y:y+h, x:x+w]
        break

    cv2.imwrite(cropped_board_path,ROI)

def predict_board():
    """
    Gets the FEN of the board
    """
    print("Start predicting:")
    fen = get_FEN(cropped_board_path, classifcation_pieces_output_path)
    print("Fen: ", fen)
    print("Getting best move!")
    get_best_move(fen)

        

def get_best_move(fen):
    """
    Gets the best move according to the predicted FEN
    """
    # importing the requests library
    import requests
  
    # api-endpoint
    BASE_URL = "http://127.0.0.1:8000/temp"
    # BASE_URL = "http://chessweb.pythonanywhere.com/temp"
    
    # defining a params dict for the parameters to be sent to the API
    PARAMS = {'f':fen}
    
    # sending get request and saving the response as response object
    r = requests.get(url = BASE_URL, params = PARAMS)
    
    # extracting data in json format
    data = r.json()
    print("Best move:")
    print(data)

def on_press(key):
    """
    Triggering a function when key is being pressed
    """
    if key == keyboard.Key.esc:
        return False  # stop listener
    try:
        k = key.char  # single-char keys
    except:
        k = key.name  # other keys
    if k in ['y']:
        print("Start working:")
        capture_screenshot(screenshot_path)
        print("took screenshot!")
        print("Locating the board from the screenshot...")
        get_board(screenshot_path)
        print("Done")
        print("Predicting FEN")
        predict_board()

    elif k in ['i']:
        print("Capturing starting position") 
        capture_screenshot(screenshot_path)
        print("took screenshot!")
        print("Locating the board from the screenshot...")
        get_board(screenshot_path)
        print("Done")
        print("Generating classifcation pieces")
        generate_pieces(cropped_board_path, classifcation_pieces_output_path)
        print("Done")

def capture_screenshot(save_path):
    """
    Captures a screenshot and saves it
    """
    snapshot = ImageGrab.grab()
    snapshot.save(save_path)

listener = keyboard.Listener(on_press=on_press)
listener.start()  # start to listen on a separate thread
print("Waiting for input")
listener.join()  # remove if main thread is polling self.keys
