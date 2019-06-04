import cv2
import numpy as np


def width(video):
    return round(video.get(3))


def height(video):
    return round(video.get(4))


def format_background(video, image):
    return image[:height(video), :width(video)]


def hue_lightness_saturation(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    hue = cv2.inRange(frame[:, :, 0], 40, 75)
    return frame, hue


def apply_mask(frame, hue):
    mask = np.zeros(frame.shape, dtype=np.uint8)
    mask[:, :, 0] = hue
    mask[:, :, 1] = hue
    mask[:, :, 2] = hue
    return cv2.GaussianBlur(mask, (11, 11), 0)


def treatment(frame):
    hls_frame, hue = hue_lightness_saturation(frame)
    return apply_mask(hls_frame, hue)


def negative_frame(frame):
    return cv2.bitwise_not(frame)


def background_frame(background, frame):
    return cv2.bitwise_and(background, frame)


def join(background, image, frame):
    without_character = negative_frame(image)
    background = background_frame(background, image)
    character = cv2.bitwise_and(frame, without_character)
    return cv2.add(background, character)


def process(video, image):
    while True:
        _, frame = video.read()
        frame = join(format_background(video, image), treatment(frame), frame)
        cv2.imshow('Processed video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # ESC
            break


if __name__ == "__main__":
    # TODO - receber imagem via parâmetro
    image = cv2.imread("image.jpg")
    # TODO - receber vídeo via parâmetro
    video = cv2.VideoCapture('videos/video.mp4')
    process(video, image)
    cv2.waitKey()
    cv2.destroyAllWindows()
