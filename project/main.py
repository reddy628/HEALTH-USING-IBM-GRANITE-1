import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load IBM Granite model
model_id = "ibm-granite/granite-3.3-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Function to ask IBM Granite model
def ask_granite(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# Main health assistant function
def health_assistant():
    print("🤖 Hello! I'm your Health AI Assistant (powered by IBM Granite).")
    name = input("What's your name? ")
    print(f"Hi {name}, please describe your symptoms (comma-separated):")
    
    symptoms_input = input("Symptoms: ")
    symptoms = [s.strip() for s in symptoms_input.split(',')]
    symptom_text = ", ".join(symptoms)
    
    print("\n🔍 Analyzing symptoms with AI...")
    prompt = f"The user reports the following symptoms: {symptom_text}. What are the most likely medical conditions or causes?"
    granite_response = ask_granite(prompt)

    print("\n📋 AI Suggested Conditions:")
    print(granite_response)

    print("\n⚠️ Note: This is not a medical diagnosis. Please consult a healthcare professional.")
    print(f"🕒 Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Run the assistant
if __name__ == "__main__":
    health_assistant()