"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""

from ex1_utils import *


def on_trackbar(val):
    pass


def gammaDisplay(img_path: str, rep: int):
    img = cv2.imread(img_path)
    if rep == LOAD_GRAY_SCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('Gamma Trackbar')
    cv2.createTrackbar('Value', 'Gamma Trackbar', 0, 100, on_trackbar)
    index = np.arange(256)
    while True:
        gamma = cv2.getTrackbarPos('Value', 'Gamma Trackbar')
        gamma = (gamma * (2 - 0.000001)) / 100.0
        if gamma == 0:
            gamma = 0.000001

        lut = ((index / 255.0) ** (1.0 / gamma)) * 255
        lut = lut.astype('uint8')
        new_img = cv2.LUT(img, lut)

        cv2.imshow('Gamma Trackbar', new_img)
        key = cv2.waitKey(1000)
        if cv2.getWindowProperty('Gamma Trackbar', cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)
    gammaDisplay('water_bear.png', LOAD_GRAY_SCALE)
    gammaDisplay('water_bear.png', LOAD_RGB)


if __name__ == '__main__':
    main()