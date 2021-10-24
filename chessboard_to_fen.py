import cv2
import numpy as np
import os

def compressed_fen(fen):
    """ 
        From: 11111q1k/1111r111/111p1pQP/111P1P11/11prn1R1/11111111/111111P1/R11111K1
        To: 5q1k/4r3/3p1pQP/3P1P2/2prn1R1/8/6P1/R5K1
        Source: https://github.com/linrock/chessboard-recognizer
    """
    for length in reversed(range(2,9)):
        fen = fen.replace(length * '1', str(length))
    return fen

def prepare_image(chessboard_img_path, save_images = False, output_path = ""):
    """ Gets a screenshot of a board and cut it into 64 tiles.
        Returns list of 64 tiles.
        Uses to prepare the classifcation pieces as well.

        Source: https://github.com/linrock/chessboard-recognizer
    """
    img_data = cv2.imread(chessboard_img_path)
    
    resized = cv2.resize(img_data, (256, 256))
    img_data = resized
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    
    chessboard_256x256_img = np.asarray(img_data, dtype=np.uint8)
    # 64 tiles in order from top-left to bottom-right (A8, B8, ..., G1, H1)
    # 64 tiles in order from top-left to bottom-right (A8, B8, ..., G1, H1)
    tiles = [None] * 64
    for rank in range(8): # rows/ranks (numbers)
        for file in range(8): # columns/files (letters)
            sq_i = rank * 8 + file
            tile = np.zeros([32, 32, 3], dtype=np.uint8)
            for i in range(32):
                for j in range(32):
                    tile[i, j] = chessboard_256x256_img[rank*32 + i,file*32 + j,]
            tiles[sq_i] = tile
            if save_images:
                original_fen = "rnbqkbnrpppppppp11111111111111111111111111111111PPPPPPPPRNBQKBNR"
                output_name = f"{original_fen[sq_i]}{rank}{file}.png"
                output_name = os.path.join(output_path, output_name)
                cv2.imwrite(output_name, tile)
    return tiles

def load_images_from_folder(folder, piece_name=""):
    """ Loads pieces from a folder"""
    images = []
    names = []
    for filename in os.listdir(folder):
        if piece_name:
            if not filename.lower().startswith(str(piece_name).lower()):
                continue
        file_name = os.path.join(folder,filename)
        image = cv2.imread(file_name)
        img = np.array(image)
        if img is not None:
            images.append(img)
            names.append(os.path.basename(file_name)[0])
    return images, names


def get_images_contours(images):
    """
    Find and keep contours for every image.
    The image is a chess piece, and every tile will be compared to it.
    Returns the contours and their area
    """
    d = []
    for i in range(len(images)):
            image = images[i]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.medianBlur(gray, 1)
            thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,3)
            cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            if len(cnts):
                current_contours = cnts[0]
                current_contour_area = cv2.contourArea(current_contours)
            else:
                current_contours = None
                current_contour_area = None

            d.append((current_contours,current_contour_area))
    return d

def get_FEN(image_path, classification_source_path):
    """ Predicts the FEN of a chessboard image
    """
    import time
    start_time = time.time()

    images, names = load_images_from_folder(classification_source_path)
    tiles = prepare_image(image_path)

    predictions = []
    images_contours = get_images_contours(images)

    #Iterate every tile in the board
    for j, tile in enumerate(tiles):
        tile_score = float('inf')
        predicted = "1" #Default: empty tile

        #Get the contours of the current tile
        tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        tile_blur = cv2.medianBlur(tile_gray, 1)
        tile_thresh = cv2.adaptiveThreshold(tile_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,7,3)
        cnts = cv2.findContours(tile_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        if len(cnts) == 0:
            # If no contours found - it is probably empty tile
            predictions.append("1")
            continue

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        tile_contours = cnts[0]
        tile_contour_area = cv2.contourArea(tile_contours)
        contour_color = [36,255,12]
        cv2.drawContours(tile,[tile_contours], 0, contour_color, 1)
        _,_,w_tile,h_tile = cv2.boundingRect(tile_contours)

        #Compare to each piece
        for i in range(len(images)):
            current_contours, current_contour_area = images_contours[i]
            if current_contours is None: #Guess it is an empty tile
                continue


            #Find the best match
            match = cv2.matchShapes(current_contours, tile_contours, 1, 0.0)
            _,_,current_w,current_h = cv2.boundingRect(current_contours)
            
            area_diff = abs(tile_contour_area - current_contour_area)
            if area_diff > 70: #Guess it is an empty tile
                continue

            if match < 1:
                mtch = str(round(match, 3))
                w_diff = abs(w_tile - current_w)
                h_diff = abs(h_tile - current_h)
                current_score = w_diff + h_diff + (area_diff/10)
                if current_score < tile_score:
                    tile_score = current_score
                    predicted = names[i]
            

        if predicted != "1":
            contour_color = [36,255,12]
            newImage = tile.copy()
            cv2.drawContours(newImage,[tile_contours], 0, contour_color, -1)
            mask = np.all(newImage == contour_color, axis=-1)
            mask = tile[mask]
            black = (np.argwhere(mask == (0,0,0)))
            white = (np.argwhere(mask == (255,255,255)))

            if len(white) > len(black) : #Some threshold
                predicted = predicted.upper()
            else:
                predicted = predicted.lower()

        predictions.append(predicted)

    print("--- %s seconds ---" % (time.time() - start_time))
    predicted_fen = compressed_fen('/'.join([''.join(r) for r in np.reshape([p[0] for p in predictions], [8, 8])]))
    print(predicted_fen)

    return predicted_fen

def generate_pieces(starting_pos_img_path, output_path):
    """
    Generates classifcation pieces from a starting position.
    @starting_pos_img_path must be a starting position.
    """
    prepare_image(starting_pos_img_path, True, output_path)


chess_position = r"<Path to an image that we want to convert to FEN>"
starting_position_img = r"<Path to an image that contains a starting position>"
output_path = r"<Path to a location that the classifcation pieces will be stored"

generate_pieces(starting_position_img, output_path)
get_FEN(chess_position, output_path)