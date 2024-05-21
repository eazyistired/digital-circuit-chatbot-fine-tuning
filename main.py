import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
import json
import transformers
import datasets
import pandas as pd


def load_config_object(config_object_path):
    with open(config_object_path) as config_file:
        return json.load(config_file)


# FIXME Remove this and find a way to import it from digital-circuit-chatbot-api
def get_tokenizer_and_model(model_path, quantization_config, device_map={"": 0}):
    tokenizer, model = None, None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        quantization_config=quantization_config,
        device_map=device_map,
    )

    assert (
        tokenizer != None and model != None
    ), f"\n \n Tokenizer or LLM not loaded for model name: {model_path} \n \n"

    return tokenizer, model


def generate_text_prompt(system_prompt: str, prompt: str, response: str) -> str:
    return f"""
    <s>[INST] <<SYS>>
        {system_prompt}
    <</SYS>>

        {prompt} [/INST]

        {response} </s>
    """.strip()


CONFIG_TYPE = "development"
SYSTEM_PROMPT = """
                Using the information contained in the context, give a comprehensive answer to the question.
                Respond only to the question asked, response should be concise and relevant to the question.
                If the answer cannot be deduced from the context, do not give an answer.
"""

if __name__ == "__main__":
    script_dir_path = os.path.dirname(__file__)
    project_dir_path = os.path.dirname(script_dir_path)

    config_object_path = os.path.join(script_dir_path, "config.json")
    configs = load_config_object(config_object_path=config_object_path)
    config = configs[CONFIG_TYPE]

    llm_model_name = config["llm_model_name"]
    fine_tuned_llm_model_name = "fine_tuned_" + llm_model_name
    fine_tunining_dataset_name = "nvidia/HelpSteer"

    print(f"\n\nConfig object: {json.dumps(config, indent=4)}\n\n")

    quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=getattr(torch, "float16"),
    #     bnb_4bit_use_double_quant=False,
    # )

    # PATH CONFIGS
    questions_database_path = os.path.join(project_dir_path, "database", "fine_tuning")
    testcases_results_folder_path = os.path.join(
        project_dir_path, "database", "results"
    )
    llm_model_path = os.path.join(project_dir_path, "models", llm_model_name)
    fine_tuned_llm_model_path = os.path.join(
        project_dir_path, "models", fine_tuned_llm_model_name
    )

    # CODE
    tokenizer, model = get_tokenizer_and_model(
        model_path=llm_model_path, quantization_config=quantization_config
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    initial_dataset = datasets.load_dataset(fine_tunining_dataset_name, split="train")
    initial_dataset_df = initial_dataset.to_pandas()
    texts_list = []
    for row in initial_dataset:
        texts_list.append(
            generate_text_prompt(
                system_prompt=SYSTEM_PROMPT,
                prompt=row["prompt"],
                response=row["response"],
            )
        )

    texts_df = pd.DataFrame(texts_list, columns=["text"])
    training_dataset = datasets.Dataset.from_pandas(texts_df)

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_params = TrainingArguments(
        output_dir="../database/fine_tuning",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=training_dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    trainer.train()

    trainer.model.save_pretrained(save_directory=fine_tuned_llm_model_path)
    trainer.tokenizer.save_pretrained(save_directory=fine_tuned_llm_model_path)
