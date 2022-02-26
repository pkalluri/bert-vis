from enum import Enum, auto

class ModelType(Enum):
	bert_base_cased = 'bert-base-cased'
	bert_large_cased = 'bert-large-cased'
	gpt2 = 'gpt2'
	gpt2_large = 'gpt2-large'

berts = [ModelType.bert_base_cased, ModelType.bert_large_cased]
gpts = [ModelType.gpt2, ModelType.gpt2_large]

def get_generic(model_type:ModelType) -> str:
	if model_type in berts: 
		return 'bert'
	elif model_type in gpts:
		return 'gpt'
# 	
# def __init__(self):
# 	    self.general = ModelGeneralType.bert if 
# 	    self.b = b
# class ModelGeneralType(Enum):
# 	bert = auto()
# 	gpt = auto()
# class ModelClass(Enum):
# 	bert = 'bert'
# 	gpt = 'gpt'
# ModelClass = Enum('ModelClass', 'bert gpt')
# def get_official(name:str) -> str:
# 	return name.replace('_','-')
# ModelType = Enum('ModelType', 'bert_base_cased bert_large_cased gpt2 gpt2_large')