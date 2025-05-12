# https://github.com/asahi417/lm-question-generation

from pprint import pprint
from lmqg import TransformersQG

# initialize model
model = TransformersQG('lmqg/t5-base-squad-qag') # QAG with End2end or Multitask Models
# model = TransformersQG(model='lmqg/t5-base-squad-qg', model_ae='lmqg/t5-base-squad-ae')  # QAG with Pipeline Models
# paragraph to generate pairs of question and answer
context = "William Turner was an English painter who specialised in watercolour landscapes. He is often known " \
          "as William Turner of Oxford or just Turner of Oxford to distinguish him from his contemporary, " \
          "J. M. W. Turner. Many of Turner's paintings depicted the countryside around Oxford. One of his " \
          "best known pictures is a view of the city of Oxford from Hinksey Hill."
# model prediction
question_answer = model.generate_qa(context)
# the output is a list of tuple (question, answer)
pprint(question_answer)

# QG only
# from pprint import pprint
# from lmqg import TransformersQG

# initialize model
model = TransformersQG(model='lmqg/t5-base-squad-qg')

# a list of paragraph
context = [
    "William Turner was an English painter who specialised in watercolour landscapes",
    "William Turner was an English painter who specialised in watercolour landscapes"
]
# a list of answer (same size as the context)
answer = [
    "William Turner",
    "English"
]
# model prediction
question = model.generate_q(list_context=context, list_answer=answer)
pprint(question)


# AE only: The QG model can be used as following. The model is the QG model.
#
# from pprint import pprint
# from lmqg import TransformersQG

# initialize model
model = TransformersQG(model='lmqg/t5-base-squad-ae')
# model prediction
answer = model.generate_a("William Turner was an English painter who specialised in watercolour landscapes")
pprint(answer)
