
!pip install transformers

from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

def answer_question(question, context):
    input_dict = tokenizer(question, context, return_tensors='tf')
    outputs = model(input_dict)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_idx = tf.argmax(start_logits, axis=1)[0].numpy()
    end_idx = tf.argmax(end_logits, axis=1)[0].numpy()
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_dict["input_ids"][0][start_idx:end_idx+1]))
    return answer

answer_question('What is my favorite color?',"I have always loved the color blue, and I think it is the best color ever--colors like red are offputting and gross")