import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

def generate_conclusions(matched_opinions):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    max_input_length = 1024 - 150  # GPT-2's max length is 1024, 150 tokens will be used for the conclusions

    conclusions = {}
    for i, (topic, opinions) in enumerate(matched_opinions.items()):
        logging.info(f"Generating conclusion for topic {i+1}/{len(matched_opinions)}")
        prompt = f"Topic: {topic}\n\nRelated Opinions:\n"
        for opinion in opinions:
            prompt += f"- {opinion}\n"
        prompt += "\nConclusion:"

        inputs = tokenizer(prompt, return_tensors="pt")

        # Truncate input_ids to fit within the max length
        if inputs.input_ids.shape[-1] > max_input_length:
            inputs.input_ids = inputs.input_ids[:, :max_input_length]

        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

        conclusion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        conclusion = conclusion.split("Conclusion:")[-1].strip()  # Extract the conclusion part
        conclusions[topic] = conclusion

    return conclusions

def save_conclusions(conclusions, file_path):
    import csv
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Topic", "Conclusion"])
        for topic, conclusion in conclusions.items():
            writer.writerow([topic, conclusion])
