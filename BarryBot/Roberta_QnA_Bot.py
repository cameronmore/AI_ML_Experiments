from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'What does BFO do?',
    'context': 'Applied ontology is a way of building data models to promote interoperability between systems and applications when passing data from one to another. Barry Smith created an upper level ontology for capturing lots of domains called Basic Formal Ontology (BFO) to do these things.'
}
res = nlp(QA_input)

# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(res)