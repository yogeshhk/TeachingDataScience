import gc
import os
import pandas as pd
from datetime import datetime
import torch
import transformers
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer
import logging


class FineTuningTrainer:
    def __init__(self,
                 pretrained_base_model="microsoft/phi-2",
                 raw_train_file_path=None,
                 formatted_train_file_path=None,
                 output_dir="./models",
                 num_epochs=1,
                 batch_size=1,
                 learning_rate=2e-4):

        self.trainer = None
        self.base_model = pretrained_base_model
        self.raw_train_file_path = raw_train_file_path
        self.formatted_train_file_path = formatted_train_file_path
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self._format_dataset()

        # Initialize components
        self._setup_tokenizer()
        self._setup_model()
        self._setup_training_args()
        self._setup_peft_config()

    def _setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def _setup_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map={'': 0}
        )

        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)

    def _setup_training_args(self):
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=128,
            eval_strategy="steps",
            eval_steps=200,
            logging_steps=50,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            learning_rate=self.learning_rate,
            save_steps=200,
            warmup_ratio=0.05,
            weight_decay=0.01,
            max_grad_norm=0.3,
            save_total_limit=2,
            fp16=True,
            report_to=["tensorboard"]
        )

    def _setup_peft_config(self):
        self.peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=["Wqkv", "fc1", "fc2"],
            bias="none"
        )

    def _format_dataset(self, raw_dataset_path=None, formatted_dataset_path=None):
        try:
            self.logger.info(f"Loading raw dataset from {raw_dataset_path}")
            dataset = load_dataset("csv", data_files=raw_dataset_path, split="train")
            df = pd.DataFrame(dataset)

            # Create default output path if none provided
            if formatted_dataset_path is None:
                dir_path = os.path.dirname(raw_dataset_path)
                base_name = os.path.splitext(os.path.basename(raw_dataset_path))[0]
                output_path = os.path.join(dir_path, f"{base_name}_formatted.csv")

            self.logger.info("Converting to LLaMA instruction format")

            def convert_to_llama_instruct_format(row):
                question = row['Context']
                answer = row['Response']
                return f"[INST] {question} [/INST] {answer} "

            df['text'] = df.apply(convert_to_llama_instruct_format, axis=1)
            formatted_df = df[['text']]  # Keep only the formatted text column

            self.logger.info(f"Saving formatted dataset to {output_path}")
            formatted_df.to_csv(output_path, index=False)

            self.train_file_path = output_path  # Update the training file path
            return output_path

        except Exception as e:
            self.logger.error(f"Error formatting dataset: {str(e)}")
            raise e

    def load_and_split_dataset(self):
        dataset = load_dataset("csv", data_files=self.train_file_path, split="train")
        split_datasets = dataset.train_test_split(test_size=0.1, seed=42)
        return split_datasets["train"], split_datasets["test"]

    def train(self):
        try:
            self.logger.info("Loading datasets...")
            train_dataset, eval_dataset = self.load_and_split_dataset()

            self.logger.info("Initializing trainer...")
            custom_trainer = SFTTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                peft_config=self.peft_config,
                tokenizer=self.tokenizer,
                args=self.training_args
            )

            self.logger.info("Starting training...")
            custom_trainer.train()

            self.trainer = custom_trainer
            return custom_trainer

        except Exception as e:
            self.logger.error(f"Training error occurred: {str(e)}")
            raise e

    def generate_text(self, prompt, max_length=250):
        try:
            pipe = transformers.pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=max_length,
                device=self.device
            )
            formatted_prompt = f"[INST] {prompt} [/INST]"
            result = pipe(formatted_prompt)
            return result[0]['generated_text']

        except Exception as e:
            self.logger.error(f"Generation error occurred: {str(e)}")
            raise e

    def cleanup(self):
        del self.model
        if hasattr(self, 'trainer'):
            del self.trainer
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    # Set your Hugging Face token
    hf_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    if hf_api_token:
        login(token=hf_api_token)
    else:
        raise ValueError("Please set HUGGINGFACEHUB_API_TOKEN environment variable")

    # Config here
    dataset_path = "D:/Yogesh/GitHub/Sarvadnya/src/fine-tuning/data/AmodMentalHealthCounselingConversations_train.csv"
    formatted_path = "D:/Yogesh/GitHub/Sarvadnya/src/fine-tuning/data" + \
                     "/AmodMentalHealthCounselingConversations_formatted.csv"
    base_model = "microsoft/phi-2"
    fine_tuned_model = "phi2-mental-health"

    # Initialize trainer with your parameters
    trainer = FineTuningTrainer(
        pretrained_base_model=base_model,
        raw_train_file_path=dataset_path,
        formatted_train_file_path=formatted_path,
        output_dir="./models/" + fine_tuned_model,
        num_epochs=1,
        batch_size=1,
        learning_rate=2e-4
    )

    # Train the model
    try:
        trainer.train()

        # Test generation
        test_prompt = "I am not able to sleep at night. Do you have any suggestions?"
        response = trainer.generate_text(test_prompt)
        print(f"\nTest Generation Result:\nPrompt: {test_prompt}\nResponse: {response}")

    finally:
        # Clean up resources
        trainer.cleanup()
