import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset

class SummarizationTrainer:
    def __init__(self, model_name: str = "t5-small"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def prepare_dataset(self, df):
        """Converts a pandas DataFrame to a HuggingFace Dataset."""
        return Dataset.from_pandas(df)

    def tokenize_function(self, examples):
        """Tokenizes the inputs and targets."""
        # For T5, prefix input with "summarize: "
        prefix = "summarize: " if "t5" in self.model_name else ""
        inputs = [prefix + doc for doc in examples["transcript"]]
        
        model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

        # Setup the tokenizer for targets
        labels = self.tokenizer(text_target=examples["summary"], max_length=128, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(self, train_df, val_df, output_dir="./results", epochs=3, batch_size=4):
        """Fine-tunes the model on the provided dataset."""
        train_dataset = self.prepare_dataset(train_df)
        val_dataset = self.prepare_dataset(val_df)

        # Tokenize datasets
        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_val = val_dataset.map(self.tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        
        # Save model
        trainer.save_model(f"{output_dir}/final_model")
        print(f"Model saved to {output_dir}/final_model")

if __name__ == "__main__":
    # Example usage:
    # from data_loader import DataLoader
    # loader = DataLoader("../data/mock_dataset.json")
    # df = loader.load_data()
    # train, val, test = loader.split_data(df)
    # trainer = SummarizationTrainer("t5-small")
    # trainer.train(train, val, epochs=1)
    pass
