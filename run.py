import argparse

import cv2

from model import Model


def argument_parser():
    parser = argparse.ArgumentParser(description="Violence detection")
    parser.add_argument('--image-path', type=str,
                        default='./data/0.jpg',
                        help='path to your image')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()
    model = Model()
    image = cv2.imread(args.image_path)
    prediction = model.predict(image=image)
    print('Prediction dict: ', prediction)
    cv2.imshow(prediction['labels'][0].title(), image)
    cv2.waitKey(0)
