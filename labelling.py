import os
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from glob import glob
from PIL import Image
import shutil


class Labelling:
    """
    Image labelling tools for area class project
    """
    labels = None
    all_file_paths = None

    def __init__(self, base_sample_dir='sample', target_sample_dir='model_sample'):
        if not os.path.exists(target_sample_dir):
            os.mkdir(target_sample_dir)
        self.target_sample_dir = target_sample_dir
        self.__base_sample_dir = base_sample_dir
        self.relabel = False

    def get_labels(self):
        if self.relabel is False:
            sample_dirs = glob(f'{self.__base_sample_dir}/*')
        else:
            sample_dirs = glob(f'{self.target_sample_dir}/*')
        labels = []
        for label in sample_dirs:
            labels.append(os.path.split(label)[-1])
        return labels

    def get_all_file_paths(self):
        result = []
        if self.relabel is False:
            for label in self.labels:
                file_paths = [os.path.join(self.__base_sample_dir, label, i) for i in
                              os.listdir(f"{self.__base_sample_dir}/{label}")]
                for file in file_paths:
                    result.append(dict(label=label, file_path=file))
        else:
            file_paths = glob(f"{self.__base_sample_dir}/*")
            for file in file_paths:
                result.append(dict(label=os.path.split(self.__base_sample_dir)[-1], file_path=file))
        return result

    def __show_sample(self, img_path, current_label):
        img = Image.open(img_path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis("off")
        plt.subplots_adjust(left=0.2)

        # Make checkbuttons with all plotted lines with correct visibility
        rax = plt.axes([0.01, 0.4, 0.3, 0.2])
        visibility = [True if i == current_label else False for i in self.labels]

        check = CheckButtons(rax, self.labels, visibility)

        # def func(label):
        #     print(f"{label} clicked")

        # check.on_clicked(func)

        plt.subplots_adjust(left=0.3)
        plt.suptitle(os.path.split(img_path)[-1])

        plt.show()

        selected_label = [self.labels[i] for i, x in enumerate(check.get_status()) if x == True]
        not_selected_label = [self.labels[i] for i, x in enumerate(check.get_status()) if x == False]
        if not selected_label:
            selected_label = ['not_classified_6_angle']
        return selected_label, not_selected_label

    def __move_file(self, src, label, filename):
        if not os.path.exists('not_classified_6_angle'):
            os.mkdir('not_classified_6_angle')
        if not os.path.exists(os.path.join(self.target_sample_dir, label)):
            os.mkdir(os.path.join(self.target_sample_dir, label))
        if label == 'not_classified_6_angle':
            target = os.path.join(label, filename)
        else:
            target = os.path.join(self.target_sample_dir, label, filename)
        if not os.path.isfile(target):
            shutil.copy(src=src, dst=target)
        print(f"{filename} copied to {target}")

    def show_all_imgs(self):

        self.labels = self.get_labels()
        self.all_file_paths = self.get_all_file_paths()

        counter = 1
        total_files = len(self.all_file_paths)
        for s in self.all_file_paths:
            file_path = s['file_path']
            current_label = s['label']
            filename = os.path.split(file_path)[-1]
            selected_label, not_selected_label = self.__show_sample(img_path=file_path, current_label=current_label)
            for label in selected_label:
                self.__move_file(src=file_path, label=label, filename=filename)
            for label in not_selected_label:
                path = f"{self.target_sample_dir}/{label}/{filename}"
                if os.path.exists(path):
                    os.remove(path)
            print(f"{counter} from {total_files} files have been processed")
            counter += 1


if __name__ == '__main__':
    # labelling = Labelling(target_sample_dir='selected_sample',
    #                       base_sample_dir='selected_sample/industrialArea')
    # labelling.relabel = True

    labelling = Labelling(target_sample_dir='selected_sample',
                          base_sample_dir='sample')

    labelling.show_all_imgs()
