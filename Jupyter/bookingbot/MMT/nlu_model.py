# # Part-1:Add Natural Language understanding(create NLU  model)

from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer, Metadata, Interpreter
from rasa_nlu import config
import pprint
import spacy
print(spacy.load("en")("hello"))

def train_nlu(data, configs, model_dir):
	training_data=load_data(data)																#load NLU training sample
	trainer = Trainer(config.load(configs))														#train the pipeline first
	interpreter = trainer.train(training_data)													#train the model
	model_directory = trainer.persist("models/nlu",fixed_model_name="chatter")					#store in directory

def run_nlu():
	interpreter = Interpreter.load('./models/nlu/default/chatter')
	pprint.pprint(interpreter.parse("CCU"))
	
if __name__ == '__main__':
	train_nlu('./data/nlu.md', './config/config.yml', './models/nlu')
	run_nlu()

