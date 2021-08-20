import cv2
import pynput
from os.path import exists, isfile, join
from os import listdir
from pathlib import Path

#  Global variables that are used across the functions
refPt = []
image = []
cropping = False
shift_held = False
window_title = 'Master Image'
resources_dir = 'resources/'
extraction_img_dir = resources_dir + 'extraction_imgs/'
positive_img_dir = resources_dir + 'pos/'
negative_img_dir = resources_dir + 'neg/'
other_saved_img_dir = resources_dir + 'saved_imgs/'


def on_press(keypress):
    global shift_held
    if keypress == pynput.keyboard.Key.shift:
        shift_held = True


def on_release(keypress):
    global shift_held
    if keypress == pynput.keyboard.Key.shift:
        shift_held = False


def format_rectangle(rectangle) -> list:
    """If the rectangle is drawn so that the second point is above or before
    the first point, the points are modified so it is a valid rectangle"""
    # print("Image Dimensions:", rectangle)
    for i in range(2):
        if rectangle[0][i] > rectangle[1][i]:
            rectangle[0][i], rectangle[1][i] = rectangle[1][i], rectangle[0][i]
    formatted_rectangle = rectangle
    return formatted_rectangle


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, image, window_title

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    # check to see if the left mouse button was released
    if event == cv2.EVENT_LBUTTONDOWN and cropping and shift_held:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append([x, y])
        cropping = False

        # draw a rectangle around the region of interest
        refPt = format_rectangle(refPt)
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 1)
        cv2.imshow(window_title, image)

    elif event == cv2.EVENT_LBUTTONDOWN and shift_held:

        refPt = [[x, y]]
        cropping = True


def extract_from_image(img_filepath: str) -> None:
    """
    This gets the specified image and allows the user to crop out an image.

    This is the main function of the program that will be processing the image.
    It gets the main image, copies the smaller portion and converts it to
    greyscale so it is ready for the AI training

    :param img_filepath: Filepath of the image to be extracted from
    :return: None
    """
    global refPt, cropping, image, window_title
    # load the image, clone it, and setup the mouse callback function
    # image = cv2.imread(args.get('image'))
    image = cv2.imread(img_filepath)
    clone = image.copy()

    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, click_and_crop)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow(window_title, image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            # if the 'r' key is pressed, reset the cropping region
            image = clone.copy()

        if len(refPt) == 2:
            # If the region of interest has been defined (with two mouse clicks)
            roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

            # Create new image and show it
            grey_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            cv2.imshow("ROI", grey_roi)

            # Check to see if the user wants to save it
            key = cv2.waitKey(0)
            save_keys = [ord('f'), ord('v'), ord('s')]
            dir_to_save = positive_img_dir if key == ord(
                'f') else negative_img_dir if key == ord(
                'v') else other_saved_img_dir
            if key in save_keys:
                i = 0
                while i < 20000:
                    file_written = False
                    file_name = "{}.jpg".format(i)
                    image_file = "{}{}".format(dir_to_save, file_name)
                    if exists(image_file):
                        i += 1
                    else:
                        cv2.imwrite(image_file, grey_roi)
                        # print("Saved:", file_name)
                        file_written = True
                        break
                    # print(i)
            image = clone.copy()
            cv2.destroyWindow("ROI")
            refPt = []
        # if the 'q' key is pressed, break from the loop
        if key == ord("q"):
            break


if __name__ == '__main__':
    print("Press SHIFT when selecting the bounds.")
    print(
        "While holding SHIFT, press 'f' to save the positive isolated image.")
    print(
        "While holding SHIFT, press 'v' to save the negative isolated image.")
    print(
        "While holding SHIFT, press 's' to save the image to the saved_imgs directory.")
    print("Press 'r' to reset and select another image.")
    print("Press 'q' to quit the current image and move onto the next.")

    # Create directories if they don't exist
    Path(extraction_img_dir[:-1]).mkdir(parents=True, exist_ok=True)
    Path(positive_img_dir[:-1]).mkdir(parents=True, exist_ok=True)
    Path(negative_img_dir[:-1]).mkdir(parents=True, exist_ok=True)
    Path(other_saved_img_dir[:-1]).mkdir(parents=True, exist_ok=True)

    listener = pynput.keyboard.Listener(on_press=on_press,
                                        on_release=on_release)
    listener.start()

    # Cycle through each of the files to crop out the desired parts
    image_files = [f for f in listdir(extraction_img_dir) if
                   isfile(join(extraction_img_dir, f))]
    for file_name in image_files:
        if file_name.endswith('.jpg'):
            extract_from_image(extraction_img_dir + file_name)

    cv2.destroyAllWindows()
    listener.stop()
