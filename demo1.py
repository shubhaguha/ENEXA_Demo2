from argparse import ArgumentParser
import json
import os
from time import time

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


print("CUDA available:", torch.cuda.is_available())     # Should return True if GPUs are available
print("CUDA device count:", torch.cuda.device_count())  # Number of GPUs available


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for line in data:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")


def write_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def create_message_column(row):
    """Create a conversational message column for the dataset."""

    # Initialize an empty list to store the messages.
    messages = []

    # Create a "system" message dictionary with "content" and "role" keys.
    system = {
        "content": "You are a helpful AI assistant specializing in Information Extraction (IE) tasks such as Named Entity Recognition (NER) or Relation Extraction (RE). Analyze the following text and provide the requested information.",
        "role": "system"
    }

    # Append the "system" message to the "messages" list.
    messages.append(system)

    # Create a "user" message dictionary with "content" and "role" keys.
    user = {
        "content": row["prompt"],
        "role": "user"
    }

    # Append the "user" message to the "messages" list.
    messages.append(user)

    # Return a dictionary with a "messages" key and the "messages" list as its value.
    return {"messages": messages}


def format_dataset_chatml(row, tokenizer):
    return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=False)}


#################################################################################################

def main():
    parser = ArgumentParser(description="Test the model with the shadowlinks dataset")
    parser.add_argument("--input_folder", type=str, default="demo_files/input_folder", help="The input folder where the dowloaded Wikipedia files are stored")
    parser.add_argument("--prompt_template", type=str, default="input_files/prompt_template_with_example.json", help="The prompt template to fill in the inputs.")
    parser.add_argument("--target_entity_types", type=str, default="input_files/target_entity_types_4_adidas.json", help="The target entity types to extract from the Wikipedia files.")
    parser.add_argument("--target_relations", type=str, default="input_files/target_relations_4_adidas.json", help="The target relations to extract from the Wikipedia files.")
    parser.add_argument("--NER_example", type=str, default="input_files/NER_example.json", help="NER example to use in the prompt template.")
    parser.add_argument("--RE_example", type=str, default="input_files/RE_example.json", help="RE example to use in the prompt template.")
    parser.add_argument("--LLM", type=str, default="FinaPolat/phi4_adaptable_IE", help="The model to use for the test")
    parser.add_argument("--temperature", type=float, default=0.001, help="The temperature to use for the test")
    parser.add_argument("--max_tokens", type=int, default=8192, help="The maximum number of tokens to use for the test")
    parser.add_argument("--output_folder", type=str, default="IE_extraction_output", help="The output folder where the resulting triples will be stored")
    args = parser.parse_args()

    args_dict = vars(args)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    template = load_json(args.prompt_template)
    target_entity_types = load_json(args.target_entity_types)
    print("Target entity type categories:", len(target_entity_types), flush=True)
    target_relations = load_json(args.target_relations)
    print("Target relation categories", len(target_relations), flush=True)
    NER_example = load_json(args.NER_example)
    RE_example = load_json(args.RE_example)

    # Create prompts
    input_data = []
    prompts = []
    start = time()
    for file in os.listdir(args.input_folder):
        if file.endswith(".json"):
            file_path = os.path.join(args.input_folder, file)
            data = load_json(file_path)
            print(f"Processing file: {file}", flush=True)
            print(f"Number of articles in {file}: {len(data)}", flush=True)
            # If the file is a Wikipedia dump, it should contain a list of articles
            # Each article is a dictionary with "url", "heading", "paragraphs", and "table" keys
            for item in data:
                if "url" in item:
                    url = item["url"]
                # NER: Named Entity Recognition
                if "paragraphs" in item:
                    merges_parags = " ".join(item["paragraphs"])
                    for t in target_entity_types:
                        text = f"article: {file.split('.')[0]}, {item['heading']}: {merges_parags}"
                        prompt = template["formatter"].format(task= "NER: Named Entity Recognition",
                                                              example= NER_example,
                                                              schema=json.dumps(t["types"], ensure_ascii=False),
                                                              inputs= text,
                                                              output_format = '[["Entity", "Type"], ...]')
                        input_data.append({"article": file.split(".")[0],
                                            "heading": item["heading"],
                                            "input type": "paragraph",
                                            "url": url,
                                            "task": "NER",
                                            "schema": t["types"],
                                            "input text": text,
                                            "prompt": prompt})
                        prompts.append({"prompt": prompt})

                if "table" in item:
                    for t in target_entity_types:
                        text = f"article: {file.split('.')[0]}, {item['heading']}: {json.dumps(item['table'], ensure_ascii=False)}"
                        prompt = template["formatter"].format(task= "NER: Named Entity Recognition",
                                                              example= NER_example,
                                                              schema=json.dumps(t["types"], ensure_ascii=False),
                                                              inputs= text,
                                                              output_format = '[["Entity", "Type"], ...]')
                        input_data.append({"article": file.split(".")[0],
                                            "heading": item["heading"],
                                            "input type": "table",
                                            "url": url,
                                            "task": "NER",
                                            "schema": t["types"],
                                            "input text": text,
                                            "prompt": prompt})
                        prompts.append({"prompt": prompt})

                # RE: Relation Extraction
                if "paragraphs" in item:
                    merges_parags = " ".join(item["paragraphs"])
                    for p in target_relations:
                        text = f"article: {file.split('.')[0]}, {item['heading']}: {' '.join(item['paragraphs'])}"
                        prompt = template["formatter"].format(task="RE: Relation Extraction",
                                                              example= RE_example,
                                                              schema=json.dumps(p["relations"], ensure_ascii=False),
                                                              inputs= text,
                                                              output_format = '[["Subject", "Relation", "Object"], ...]')
                        input_data.append({"article": file.split(".")[0],
                                           "heading": item["heading"],
                                           "input type": "paragraph",
                                           "url": url,
                                           "task": "RE",
                                           "schema": p["relations"],
                                           "input text": text,
                                           "prompt": prompt})
                        prompts.append({"prompt": prompt})

                if "table" in item:
                    for p in target_relations: 
                        text = f"article: {file.split('.')[0]}, {item['heading']}: {json.dumps(item['table'], ensure_ascii=False)}"  
                        prompt = template["formatter"].format(task="Relation Extraction",
                                                              example= RE_example,
                                                              schema=json.dumps(p["relations"], ensure_ascii=False),
                                                              inputs= text,
                                                              output_format = '[["Subject", "Relation", "Object"], ...]')
                        input_data.append({"article": file.split(".")[0],
                                           "heading": item["heading"],
                                           "input type": "table",
                                           "url": url,
                                           "task": "RE",
                                           "schema": p["relations"],
                                           "input text": text,
                                           "prompt": prompt})
                        prompts.append({"prompt": prompt})
    print(f"Prompts created ({time() - start}s):", len(input_data), flush=True)

    # Load dataset
    start = time()
    dataset = Dataset.from_list(prompts)
    print("Data for inference is loaded:", len(dataset), flush=True)
    # print(dataset[0], flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.LLM, padding_side="left")
    data = dataset.map(create_message_column)
    data = data.map(format_dataset_chatml, fn_kwargs={"tokenizer": tokenizer})
    columns_to_remove = ["messages", "prompt"]
    data = data.map(lambda x: x, remove_columns=columns_to_remove)
    print(f"Data for inference is formatted ({time() - start}s):", len(dataset), flush=True)
    # print(data[0], flush=True)

    max_seq_length = args.max_tokens
    print("=== HERE ===")
    start = time()
    model = AutoModelForCausalLM.from_pretrained("FinaPolat/phi4_adaptable_IE", low_cpu_mem_usage=True, resume_download=True)
    print("=== DONE ===")
    print(f"Model is loaded ({time() - start}s)")
    start = time()
    tokenizer = AutoTokenizer.from_pretrained("FinaPolat/phi4_adaptable_IE")
    print(f"Tokenizer is loaded ({time() - start}s)")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # Add to tokenizer
    tokenizer.padding_side = "left"
    model.resize_token_embeddings(len(tokenizer))

    LLM_answers = []
    for i in range(len(data)):
        print(f"Generating answer for row {i}...", flush=True)
        print(data[i]["text"], flush=True)
        inputs = tokenizer(data[i]["text"], return_tensors="pt").to("cuda")
        outputs = model.generate(input_ids=inputs["input_ids"], 
                                 attention_mask=inputs["attention_mask"], 
                                 max_new_tokens = args.max_tokens,
                                 temperature=args.temperature,
                                 do_sample=True)
        prompt_length = inputs["input_ids"].shape[1]
        generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        row = input_data[i]
        LLM_answers.append({
            "article": row["article"],
            "heading": row["heading"],
            "url": row["url"],
            "input type": row["input type"],
            "task": row["task"],
            "schema": row["schema"],
            "input text": row["input text"],
            "prompt": row["prompt"],
            "LLM answer": generated_text,
        })

    write_jsonl(LLM_answers, f"{args.output_folder}/LLM_answers.jsonl")
    write_json(args_dict, f"{args.output_folder}/args.json")


if __name__ == "__main__":
    main()
