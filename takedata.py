import cv2
import os

print ("Enter the number of sample images you want to take")
num_of_images = int(input())
print ("Enter the label of the images you want to take")
image_label = input()

# All the captured categories will have a sub-folder placed inside the 'training_img_folder'
training_img_folder = 'training_images'

# The label_name is the name of our category (Eg: - up, down, chrome etc.)
label_name = os.path.join(training_img_folder, image_label)

# Starting image file number
image_name = 0

try:
    os.mkdir(training_img_folder)
except FileExistsError:
    pass

try:
    os.mkdir(label_name)
except FileExistsError:
    # If any images are already present, updating the image name starting number
    image_name = len(os.listdir(label_name))

# Begin Capturing Images from webcam
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)


# count of images to be captured
count = 0
font = cv2.FONT_HERSHEY_PLAIN

# Flag to check if user has clicked 's' or not, initially set to False
click = False

while True:
    ret, image = video.read()

    # Flip image horizontally to correct the mirror image
    image = cv2.flip(image, 1)

    # Stop capturing images once the count is reached
    if count == num_of_images:
        break

    # Drawing a square with white border. Anything inside this square box will be captured as training image.
    cv2.rectangle(image, (200, 200), (550, 550), (255, 255, 255), 2)

    # check the key pressed
    k = cv2.waitKey(1)
    if k == 13 or k == ord('q'):
        break
    if k == ord('s'):
        click = True

    # Start capturing pictures when user presses 's' key
    if click:
        region_of_interest = image[200:550, 200:550]

        # form the location of the image to be saved
        image_name = image_name + 1
        save_path = os.path.join(label_name, '{}.jpg'.format(image_name))

        # save image at the location
        cv2.imwrite(save_path, region_of_interest)

        count = count + 1
        click = False

    # putText() method is used here to display message inside the webcam feed. It takes the following parameters
    # <image> : the image where the text is to be displayed
    # <text> : text to be displayed
    # (x,y) : position of the text
    # <font> : the font name of the text
    # <font_size>: size of the font
    # (BGR) : the color of the text in BGR format
    # <font_thickness> : thickness of the text characters
    cv2.putText(image, "Fit the gesture inside the white box and Press 's' key to start clicking pictures",
                (20, 30), font, 1.5, (12, 20, 200), 2)
    cv2.putText(image, "Press 'q' or enter to exit.",
                (20, 60), font, 1.5, (12, 20, 200), 2)
    cv2.putText(image, "Image Count: {}".format(count),
                (20, 100), font, 1.5, (12, 20, 200), 2)
    cv2.imshow("Get Training Images", image)

print("\n\nDone\n\n")
video.release()
cv2.destroyAllWindows()
