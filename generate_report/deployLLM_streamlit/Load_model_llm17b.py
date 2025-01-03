from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import streamlit as st
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def load_model():
    model_name = "ar08/Llama-3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer

classifier = load_classifier()
model, tokenizer = load_model()

st.title("LLM Fine-tune")

user_input = st.text_area("Enter your question:")


def is_biotech_related(question, max_length=500):
    """
    Check if the question is related to biotechnology.
    Use zero-shot-classification model for classification.
    Returns True if relevant, False if unrelated.
    """
    candidate_labels = ["biotechnology", "not related to biotechnology"]

    result = classifier(question, candidate_labels)

    label = result['labels'][0]

    # Return True if the label is "biotechnology", otherwise False
    return label == "biotechnology"

def llama_generate_answer(question, max_length=500):
    if not is_biotech_related(question):
        return "This question is unrelated, please try again!"
    else:
        prompt = (
            "You are an expert biotechnology consultant. Answer the following question with precision, clarity, and depth. "
            "Provide a structured, detailed response that demonstrates in-depth knowledge of biotechnology products and techniques.\n\n"
            "Guidelines:\n"
            "- Be specific and technical\n"
            "- Use precise scientific terminology\n"
            "- Reference specific product details when possible\n"
            "- Explain the rationale behind your recommendation\n\n"
            "Please answer truthfully and write out your "
            "thinking step by step to be sure you get the right answer. If you make a mistake or encounter "
            "an error in your thinking, say so out loud and attempt to correct it. If you don't know or "
            "aren't sure about something, say so clearly.\n\n"
            "Question: {question}\n\n"
            "Answer: <<ANSWER>>".format(question=question)
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=True,
        max_length=1000,
    ).to(device)

    output = model.generate(
        inputs['input_ids'],
        attention_mask=inputs["attention_mask"],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_length=500,
        no_repeat_ngram_size=4,
        temperature=0.3,
        # early_stopping=True,
        top_k=60,
        top_p=0.55,
        do_sample=True,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True).split("<<ANSWER>>")[-1].strip()

if st.button("Generate"):
    if not user_input.strip():
        st.write("## Please enter a valid question.")
    elif not is_biotech_related(user_input):
        st.write("## This question is unrelated, please try again!")
    else:
        response = llama_generate_answer(user_input)
        st.write(f"## Result:\n{response}")