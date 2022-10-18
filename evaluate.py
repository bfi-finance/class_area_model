import os
import tensorflow as tf
import pickle as pk
from glob import glob
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image
from pprint import pprint
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval


class Evaluate:

    def __init__(self, sample_dir,
                 threshold,
                 model_src_path='model_src'):
        self.model_src_path = model_src_path
        self.sample_dir = sample_dir
        with open(os.path.join(self.model_src_path, 'label.pk'), 'rb') as l:
            self.label = pk.load(l)
        self.threshold = threshold * 100

    def single_predict(self, input_path):
        model = tf.keras.models.load_model(os.path.join(self.model_src_path, 'my_model'), compile=True)
        # model = tf.keras.models.load_weights(self.model_src_path)
        img_input = image.load_img(input_path, target_size=(224, 224, 3))
        img = image.img_to_array(img_input)
        img = np.expand_dims(img, 0)
        pred = model.predict(img, verbose=0).flatten()
        predictions = [round(float(i) * 100, 2) for i in pred]
        return {c: p for c, p in zip(self.label, predictions)}

    def batch_predict(self):
        model = tf.keras.models.load_model(os.path.join(self.model_src_path, 'my_model'), compile=True)
        df = self.get_sample_df()
        batch_holder = np.zeros((df.shape[0], 224, 224, 3))
        img_dir = df.file.tolist()
        for i, img in enumerate(img_dir):
            img = image.load_img(img, target_size=(224, 224))
            img = image.img_to_array(img)
            batch_holder[i, :] = img
        pred = model.predict(batch_holder, verbose=0)
        pred = [list(i) for i in pred]
        get_label = lambda x: [i[0] for i in zip(self.label, x) if i[1]*100 >= self.threshold]
        predictions = [get_label(x=i) for i in pred]
        predictions = pd.Series(predictions)
        return predictions

    def __get_label_predict(self, prediction):
        label = list(prediction.keys())
        val = list(prediction.values())

        idx = [i[0] for i in enumerate(val) if i[1] >= self.threshold]
        result = [i[1] for i in enumerate(label) if i[0] in idx]
        # print(val)
        # print(', '.join(result))
        return result

        # max_val = max(val)
        # if max_val > 0.3:
        #     idx = val.index(max_val)
        #     predicted_label = label[idx]
        # else:
        #     predicted_label = np.NaN
        # return predicted_label

    def get_sample_df(self):
        sample_dir = os.listdir(self.sample_dir)
        result = []
        for label in sample_dir:
            label_dir = os.path.join(dir, label)
            files = os.listdir(label_dir)
            for f in files:
                file_path = os.path.join(label_dir, f)
                file_path = os.path.normpath(file_path)
                result.append(dict(label=label, file=file_path))
        result_df = pd.DataFrame(result)
        return result_df

    def predict_sample(self):
        df = self.get_sample_df()
        # df['predicted_label'] = df.file.apply(
        #     lambda x: self.__get_label_predict(prediction=self.single_predict(input_path=x)))
        df['predicted_label'] = self.batch_predict()
        return df

    def get_confmat(self):
        sample_df = self.get_sample_df()
        pred_df = self.predict_sample()
        sample_df = sample_df.pivot_table(index=['file'], columns='label', aggfunc=len, fill_value=0). \
            reset_index()
        # pred_df.predicted_label = pred_df.predicted_label.apply(lambda x: literal_eval(x))
        pred_csv = pred_df[['file', 'predicted_label']].explode('predicted_label')
        pred_csv = pred_csv.pivot_table(index=['file'], columns='predicted_label', aggfunc=len, fill_value=0). \
            reset_index()
        cols = [i for i in self.label if i not in pred_csv.columns and i != 'file']
        for col in cols:
            pred_csv[col] = 0

        pred_csv = pred_csv.rename(columns={i: f'predicted_{i}' for i in self.label})
        pred_csv = sample_df.merge(pred_csv, on=['file'], how='left').fillna(value=0)
        true_value = pred_csv[self.label]
        pred_value = pred_csv[[f'predicted_{i}' for i in self.label]]
        eval = classification_report(
            true_value,
            pred_value,
            output_dict=False,
            target_names=self.label
        )
        return eval

    def get_evaluation(self):
        pred_df = self.predict_sample()
        pred_df.to_csv('pred_df.csv', index=False)
        pred_df['true'] = np.where(pred_df.label == pred_df.predicted_label,
                                   1,
                                   0)
        accuracy = round((pred_df[pred_df['true'] == 1].shape[0] / pred_df.shape[0]) * 100, 2)
        matrix_df = pred_df.label.value_counts().reset_index().rename(columns={'index': 'label', 'label': 'total'})
        true_pred_df = pred_df[pred_df['true'] == 1].label.value_counts().reset_index().rename(
            columns={'index': 'label', 'label': 'total_true'})
        matrix_df = matrix_df.merge(true_pred_df, on=['label'], how='left')
        matrix_df['accuracy'] = matrix_df[['total_true', 'total']].apply(lambda x: round(x[0] / x[1] * 100, 2), axis=1)
        matrix_df = matrix_df[['label', 'accuracy']]
        matrix_df['label_accuracy'] = matrix_df[['label', 'accuracy']].apply(lambda x: f"{x[0]} {x[1]}%", axis=1)

        y_true = np.array(pred_df.label.tolist())
        y_pred = np.array(pred_df.predicted_label.tolist())
        label = pred_df.label.unique().tolist()
        cm = confusion_matrix(y_true, y_pred)

        col_labels = ['label', 'accuracy']
        table_vals = matrix_df[['label', 'accuracy']].values
        the_table = plt.table(cellText=table_vals,
                              colWidths=[0.2] * 2,
                              # rowLabels=row_labels,
                              # cellColours=colors,
                              colLabels=col_labels,
                              loc='center',
                              cellLoc='center')

        the_table.auto_set_font_size(False)
        the_table.set_fontsize(7)
        the_table.scale(1.5, 1.5)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=label)
        disp.plot(cmap=plt.cm.Blues)
        plt.suptitle(f"set: {self.sample_dir}")
        plt.title(f"accuracy: {accuracy}")
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    dir = 'model_sample/validation'
    pred = Evaluate(sample_dir=dir,
                    threshold=0.5)
    print(pred.get_confmat())
