import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image


class Model:
    def __init__(self, settings_path: str = './settings.yaml'):
        with open(settings_path, "r") as file:
            self.settings = yaml.safe_load(file)

        self.device = self.settings['model-settings']['device']
        self.model_name = self.settings['model-settings']['model-name']
        self.threshold = self.settings['model-settings']['prediction-threshold']
        self.top_k = self.settings['model-settings']['top_k']
        self.model, self.preprocess = clip.load(self.model_name,
                                                device=self.device)
        self.labels = self.settings['label-settings']['labels']
        self.labels_ = []
        for label in self.labels:
            text = 'a photo of ' + label  # will increase model's accuracy
            self.labels_.append(text)

        self.text_features = self.vectorize_text(self.labels_)
        self.default_label = self.settings['label-settings']['default-label']

    @torch.no_grad()
    def transform_image(self, image: np.ndarray):
        pil_image = Image.fromarray(image).convert('RGB')
        tf_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        return tf_image

    @torch.no_grad()
    def tokenize(self, text: list):
        text = clip.tokenize(text).to(self.device)
        return text

    @torch.no_grad()
    def vectorize_text(self, text: list):
        tokens = self.tokenize(text=text)
        text_features = self.model.encode_text(tokens)
        return text_features

    @torch.no_grad()
    def softmax(self, input_tensor: torch.Tensor, t: float = 0.05):
        logits = torch.exp(input_tensor / t)
        summation = torch.sum(logits, axis=0)
        z = logits / summation
        return z

    @torch.no_grad()
    def predict_(self, text_features: torch.Tensor,
                 image_features: torch.Tensor):
        # Pick the top_k most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        values, indices = similarity[0].topk(self.top_k)
        values = self.softmax(values)
        return values, indices

    @torch.no_grad()
    def predict(self, image: np.array) -> dict:
        '''
        Does prediction on an input image

        Args:
            image (np.array): numpy image with RGB channel ordering type.
                              Don't forget to convert image to RGB if you
                              read images via opencv, otherwise model's accuracy
                              will decrease.

        Returns:
            (dict): dict that contains predictions:
                    {
                    'label': 'some_label',
                    'confidence': 0.X
                    }
                    confidence is calculated based on cosine similarity,
                    thus you may see low conf. values for right predictions.
        '''
        tf_image = self.transform_image(image)
        image_features = self.model.encode_image(tf_image)
        values, indices = self.predict_(text_features=self.text_features,
                                        image_features=image_features)

        max_pred_conf = abs(values[0].cpu().item())
        if max_pred_conf < self.threshold:
            prediction = {
                'labels': [self.default_label],
                'confidences': [max_pred_conf]
            }
            return prediction

        label_indices = indices.cpu().numpy()
        confidence_scores = values.cpu().numpy().tolist()
        predicted_labels = [self.labels[i] for i in label_indices]
        prediction = {
            'labels': predicted_labels,
            'confidences': confidence_scores
        }
        return prediction

    @staticmethod
    def plot_image(image: np.array, title_text: str):
        plt.figure(figsize=[13, 13])
        plt.title(title_text)
        plt.axis('off')
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)


if __name__ == '__main__':
    model = Model()
    image = cv2.imread('./data/0.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    best_match_label = model.predict(image=image)['labels'][0]
    print('Image label is: ', best_match_label)
