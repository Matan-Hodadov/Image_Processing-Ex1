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
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    img = cv2.imread(img_path)
    if rep == LOAD_GRAY_SCALE:  # for gray image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # window_title = 'Gamma Display'
    # trackbar_name = 'Gamma'
    cv2.namedWindow('Gamma Display')
    cv2.createTrackbar('Value', 'Gamma Display', 0, 100, on_trackbar)
    while True:
        gamma = cv2.getTrackbarPos('Value', 'Gamma Display')
        # gamma = gamma/100 * (2 - 0.01)
        gamma = (gamma * (2 - 0.001)) / 100.0
        # gamma = 0.01 if gamma == 0 else gamma
        if gamma == 0:
            gamma = 0.01
        new_img = adjust_gamma(img, gamma)
        cv2.imshow('Gamma Display', new_img)
        k = cv2.waitKey(1000)
        if cv2.getWindowProperty('Gamma Display', cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def adjust_gamma(org_image: np.ndarray, gamma: float) -> np.ndarray:
    index = np.arange(256)
    lut = ((index / 255.0) ** (1.0 / gamma)) * 255
    lut = lut.astype('uint8')
    return cv2.LUT(org_image, lut)

def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)
    gammaDisplay('water_bear.png', LOAD_GRAY_SCALE)
    gammaDisplay('water_bear.png', LOAD_RGB)


if __name__ == '__main__':
    main()