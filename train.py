import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task.translation import Translation


raw_path = './/data//'
config_path = './/dict//'

if __name__ == '__main__':
    translation = Translation(
        data_file=raw_path + 'data.txt',
        saved_dict=config_path + 'Tokenizer.txt',
        preload_file=raw_path + 'preload_data.bin',
        model_file=config_path + 'curr_best_model.txt',
    )

    translation.train(epochs=100, batch_size=128)
