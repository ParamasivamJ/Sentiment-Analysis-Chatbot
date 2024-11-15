import openvino as ov
from transformers import AutoTokenizer, AutoConfig, DataCollatorWithPadding
from optimum.intel.openvino import OVModelForCausalLM, OVModelForSequenceClassification, OVQuantizer
import gradio as gr
from uuid import uuid4
from threading import Event, Thread
import torch
import warnings
from transformers import pipeline

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define models and model paths
model_name_chat = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
model_name_sentiment = "distilbert-base-uncased-finetuned-sst-2-english"

# Load models using OpenVINO
ov_model_chat = OVModelForCausalLM.from_pretrained(model_name_chat, export=True, compile=False)
ov_model_sentiment = OVModelForSequenceClassification.from_pretrained(model_name_sentiment, export=True, compile=False)

# Load tokenizers
ov_model_chat_tok = AutoTokenizer.from_pretrained(model_name_chat, trust_remote_code=False)
ov_model_sentiment_tok = AutoTokenizer.from_pretrained(model_name_sentiment, trust_remote_code=False)

# Apply dynamic padding for better performance
data_collator_chat = DataCollatorWithPadding(tokenizer=ov_model_chat_tok, padding="longest")
data_collator_sentiment = DataCollatorWithPadding(tokenizer=ov_model_sentiment_tok, padding="longest")

# Check if quantized models exist, otherwise quantize
quantized_model_chat_path = "model/model_chat/quantized"
quantized_model_sentiment_path = "model/model_sentiment/quantized"

# Load quantized models if available, else quantize and save
if not os.path.exists(quantized_model_chat_path):
    quantizer_chat = OVQuantizer.from_pretrained(ov_model_chat)
    quantized_model_chat = quantizer_chat.quantize(export=True, optimization_config={"approach": "dynamic"}, save_directory="model/model_chat")
if not os.path.exists(quantized_model_sentiment_path):
    quantizer_sentiment = OVQuantizer.from_pretrained(ov_model_sentiment)
    quantized_model_sentiment = quantizer_sentiment.quantize(export=True, optimization_config={"approach": "dynamic"}, save_directory="model/model_sentiment")

# Reload quantized models for performance
ov_model_chat = OVModelForCausalLM.from_pretrained(quantized_model_chat_path, device="CPU")
ov_model_sentiment = OVModelForSequenceClassification.from_pretrained(quantized_model_sentiment_path, device="CPU")

# Sentiment analysis function
pipe = pipeline("text-classification", model=ov_model_sentiment, tokenizer=ov_model_sentiment_tok)

def get_sentiment(text):
    outputs = pipe(text)
    sentiment = outputs[0]["label"]
    sentiment_score = outputs[0]["score"]
    if sentiment == "NEGATIVE":
        sentiment_score *= -1
    return sentiment_score

# Chatbot response generation
def get_sentiment_label(score):
    if score > 0.5:
        return "Positive"
    elif score < -0.5:
        return "Negative"
    else:
        return "Neutral"

def bot(history, temperature, top_p, top_k, repetition_penalty, conversation_id):
    user_message = history[-1][0]
    user_sentiment = get_sentiment(user_message)
    sentiment_label = get_sentiment_label(user_sentiment)

    # Chat history formatting
    messages = "".join([f"<human>: {msg[0]} <bot>: {msg[1]}\n" for msg in history])

    input_ids = ov_model_chat_tok(messages, return_tensors="pt").input_ids
    streamer = TextIteratorStreamer(ov_model_chat_tok, timeout=30.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": 256,
        "temperature": temperature,
        "do_sample": temperature > 0.0,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "streamer": streamer,
    }

    def generate_and_signal_complete():
        ov_model_chat.generate(**generate_kwargs)

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        history[-1][1] = f"**USER_SENTIMENT: {sentiment_label}**\n" + partial_text
        yield history

# Gradio Interface
def get_uuid():
    return str(uuid4())

with gr.Blocks() as demo:
    conversation_id = gr.State(get_uuid)
    gr.Markdown("# Sentiment Analysis Chatbot")
    chatbot = gr.Chatbot(height=800)

    with gr.Row():
        msg = gr.Textbox(placeholder="Type your message here", show_label=False)
        submit = gr.Button("Submit")
        stop = gr.Button("Stop")
        clear = gr.Button("Clear")

    submit_event = msg.submit(fn=bot, inputs=[msg, chatbot], outputs=chatbot)
    submit.click(fn=bot, inputs=[msg, chatbot], outputs=chatbot)
    stop.click(fn=None, inputs=None, outputs=None)
    clear.click(lambda: None, None, chatbot)

demo.launch(server_name="0.0.0.0", server_port=8000)
