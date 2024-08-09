from datasets import load_dataset
import json

hf_dataset = "verifiers-for-code/plan500"

system_prompt = """You are given the start of a function for a Python program. Your job is to produce a detailed plan. First, analyze and think about the function, then produce a plan. Do not generate any code. The function and docstring will be provided, so they do not need to be defined or initialized again within your plan.

Respond in the following format:

<thinking>
Your thought process and analysis of the function goes here. This should include considerations about the function's purpose, inputs, outputs, and any potential challenges or considerations.
</thinking>

<plan>
Your detailed plan for implementing the function goes here. This should outline the steps to implement the function without including actual code.
</plan>

Ensure your response follows this exact format, with the analysis enclosed in <thinking> tags and the plan enclosed in <plan> tags. The content within each tag should be a continuous paragraph without line breaks."""

df = load_dataset(hf_dataset)

print(df)

# Open a file to write the JSONL output
with open("./data/codegen500k.json", "w") as f:
    # Iterate through the dataset
    for item in df["train"]:  # Assuming you're using the "test" split
        # Create three separate JSON items for each input
        json_item = {
            "instruction" : system_prompt,
            "input" : item["input"],
            "output" : item["plan"]
        }
        
        f.write(json.dumps(json_item) + "\n")

print("Conversion complete. Output written to output.jsonl")
