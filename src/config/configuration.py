from dataclasses import dataclass
import yaml

@dataclass
class DataIngestionConfig:
    source_path: str
    raw_data_path: str
    train_data_path: str
    test_data_path: str
    test_size: float
    random_state: int

@dataclass
class ModelTrainingConfig:
    image_width: int
    image_height: int
    batch_size: int
    num_class: int
    epochs: int
    fine_tuning_epochs: int
    lr: float
    save_path: str

class ConfigurationManager:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        return DataIngestionConfig(
            source_path=self.config["data_ingestion"]["source_path"],
            raw_data_path=self.config["data_ingestion"]["raw_data_path"],
            train_data_path=self.config["data_ingestion"]["train_data_path"],
            test_data_path=self.config["data_ingestion"]["test_data_path"],
            test_size=self.config["data_ingestion"]["test_size"],
            random_state=self.config["data_ingestion"]["random_state"]
        )

    def get_model_training_config(self) -> ModelTrainingConfig:
        return ModelTrainingConfig(
            image_width=self.config["model_training"]["image_width"],
            image_height=self.config["model_training"]["image_height"],
            batch_size=self.config["model_training"]["batch_size"],
            num_class=self.config["model_training"]["num_class"],
            epochs=self.config["model_training"]["epochs"],
            fine_tuning_epochs=self.config["model_training"]["fine_tuning_epochs"],
            lr=self.config["model_training"]["lr"],
            save_path=self.config["model_training"]["save_path"]
        )
