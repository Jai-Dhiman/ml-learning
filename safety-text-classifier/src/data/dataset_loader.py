"""
Safety Text Classification Dataset Loader

This module provides functionality to load and preprocess safety classification datasets
from HuggingFace and other sources for the Constitutional AI research project.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import yaml

logger = logging.getLogger(__name__)


class SafetyDatasetLoader:
    """
    Loads and preprocesses safety classification datasets for training.
    
    Supports multiple dataset sources and provides unified preprocessing
    for the safety text classification task.
    """
    
    def __init__(self, config_path: str = "configs/base_config.yaml"):
        """
        Initialize the dataset loader with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.data_config['tokenizer']
        )
        
        # Safety categories mapping
        self.safety_categories = {
            'hate_speech': 0,
            'self_harm': 1, 
            'dangerous_advice': 2,
            'harassment': 3
        }
        
    def load_toxic_bert_dataset(self) -> Dataset:
        """Load and preprocess the toxic-bert dataset."""
        logger.info("Loading toxic-bert dataset...")
        
        try:
            dataset = load_dataset("unitary/toxic-bert")
            # Process the dataset to match our safety categories
            return self._process_toxic_bert(dataset)
        except Exception as e:
            logger.warning(f"Failed to load toxic-bert dataset: {e}")
            return None
    
    def load_toxic_comment_dataset(self) -> Dataset:
        """Load and preprocess the toxic comment dataset."""
        logger.info("Loading toxic comment dataset...")
        
        try:
            # For demo purposes, we'll create a synthetic dataset
            # In practice, you'd load from the actual dataset
            texts = [
                "You are such an idiot, nobody likes you",  # hate_speech
                "I want to hurt myself because I'm worthless",  # self_harm
                "Here's how to make dangerous explosives at home",  # dangerous_advice
                "I'm going to find where you live and harass you",  # harassment
                "This is a normal, safe message about weather",  # safe
                "I love spending time with my family",  # safe
            ]
            
            labels = [
                [1, 0, 0, 0],  # hate_speech
                [0, 1, 0, 0],  # self_harm
                [0, 0, 1, 0],  # dangerous_advice
                [0, 0, 0, 1],  # harassment
                [0, 0, 0, 0],  # safe
                [0, 0, 0, 0],  # safe
            ]
            
            return Dataset.from_dict({
                'text': texts,
                'labels': labels
            })
            
        except Exception as e:
            logger.warning(f"Failed to load toxic comment dataset: {e}")
            return None
    
    def _process_toxic_bert(self, dataset: DatasetDict) -> Dataset:
        """Process the toxic-bert dataset to match our format."""
        # This is a placeholder - you'd implement actual processing logic
        # based on the specific structure of the toxic-bert dataset
        
        def process_example(example):
            # Map original labels to our safety categories
            # This would need to be adapted based on actual dataset structure
            labels = [0, 0, 0, 0]  # Initialize as safe
            
            # Example processing logic (adapt based on actual dataset)
            if 'toxic' in example and example.get('toxic', 0) > 0.5:
                labels[0] = 1  # hate_speech
            
            return {
                'text': example.get('comment_text', ''),
                'labels': labels
            }
        
        if isinstance(dataset, DatasetDict):
            processed = {}
            for split_name, split_data in dataset.items():
                processed[split_name] = split_data.map(process_example)
            return concatenate_datasets(list(processed.values()))
        else:
            return dataset.map(process_example)
    
    def create_synthetic_dataset(self, size: int = 1000) -> Dataset:
        """
        Create a synthetic dataset for testing and development.
        
        Args:
            size: Number of examples to generate
            
        Returns:
            Dataset with synthetic safety classification examples
        """
        logger.info(f"Creating synthetic dataset with {size} examples...")
        
        # Templates for different safety categories
        templates = {
            'hate_speech': [
                "I hate {group} people, they are terrible",
                "{group} are the worst, they should not exist",
                "All {group} are stupid and worthless",
            ],
            'self_harm': [
                "I want to hurt myself by {method}",
                "Instructions for self-harm: {method}",
                "How to cause pain to yourself using {method}",
            ],
            'dangerous_advice': [
                "Here's how to make {dangerous_item} at home",
                "Instructions for creating {dangerous_item}",
                "Easy recipe for {dangerous_item}",
            ],
            'harassment': [
                "I'm going to find you and {threat}",
                "You better watch out, I will {threat}",
                "I know where you live, I will {threat}",
            ],
            'safe': [
                "The weather is nice today",
                "I enjoy reading books in my free time",
                "Cooking is a wonderful hobby to have",
                "Exercise is important for good health",
                "I love spending time with friends and family",
            ]
        }
        
        # Fill-in values for templates
        groups = ["women", "men", "teenagers", "elderly"]
        methods = ["cutting", "burning", "hitting"]
        dangerous_items = ["explosives", "poison", "weapons"]
        threats = ["hurt you", "make you pay", "teach you a lesson"]
        
        texts = []
        labels = []
        
        # Generate examples for each category
        examples_per_category = size // 5  # 5 categories including safe
        
        for category, category_templates in templates.items():
            for _ in range(examples_per_category):
                template = np.random.choice(category_templates)
                
                # Fill in template based on category
                if category == 'hate_speech':
                    text = template.format(group=np.random.choice(groups))
                elif category == 'self_harm':
                    text = template.format(method=np.random.choice(methods))
                elif category == 'dangerous_advice':
                    text = template.format(dangerous_item=np.random.choice(dangerous_items))
                elif category == 'harassment':
                    text = template.format(threat=np.random.choice(threats))
                else:  # safe
                    text = template
                
                texts.append(text)
                
                # Create one-hot encoded labels
                label = [0, 0, 0, 0]
                if category != 'safe':
                    label[self.safety_categories[category]] = 1
                labels.append(label)
        
        return Dataset.from_dict({
            'text': texts,
            'labels': labels,  # Keep multi-hot encoded labels
            'source': ['synthetic'] * len(texts)
        })
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """
        Tokenize the text data using the configured tokenizer.
        
        Args:
            dataset: Dataset to tokenize
            
        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=self.data_config['max_length'],
                return_tensors=None
            )
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
    
    def create_data_splits(
        self, 
        dataset: Dataset
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            dataset: Complete dataset to split
            
        Returns:
            Tuple of (train, val, test) datasets
        """
        train_split = self.data_config['train_split']
        val_split = self.data_config['val_split']
        test_split = self.data_config['test_split']
        
        # First split: separate test set
        split_1 = dataset.train_test_split(
            test_size=test_split,
            seed=42
        )
        train_val = split_1['train']
        test = split_1['test']
        
        # Second split: separate train and validation
        val_size_adjusted = val_split / (train_split + val_split)
        split_2 = train_val.train_test_split(
            test_size=val_size_adjusted,
            seed=42
        )
        train = split_2['train']
        val = split_2['test']
        
        logger.info(f"Data splits created: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        
        return train, val, test
    
    def load_and_prepare_data(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Main method to load, process, and prepare all datasets.
        
        Returns:
            Tuple of (train, val, test) datasets ready for training
        """
        logger.info("Starting data loading and preparation...")
        
        # Load datasets from different sources
        datasets_to_combine = []
        
        # Try to load datasets from config
        for dataset_info in self.data_config['datasets']:
            try:
                if isinstance(dataset_info, str):
                    # Handle old format (just string)
                    dataset_name = dataset_info
                    config_name = None
                else:
                    # Handle new format (dict with name and config)
                    dataset_name = dataset_info['name']
                    config_name = dataset_info.get('config')
                
                logger.info(f"Loading dataset: {dataset_name} (config: {config_name})")
                
                if config_name:
                    dataset = load_dataset(dataset_name, config_name)
                else:
                    dataset = load_dataset(dataset_name)
                    
                processed_dataset = self._process_huggingface_dataset(dataset, dataset_name)
                if processed_dataset:
                    datasets_to_combine.append(processed_dataset)
            except Exception as e:
                logger.warning(f"Failed to load dataset {dataset_name}: {e}")
        
        # Create synthetic data for development
        synthetic = self.create_synthetic_dataset(size=2000)
        datasets_to_combine.append(synthetic)
        
        # Combine all datasets
        if len(datasets_to_combine) > 1:
            combined_dataset = concatenate_datasets(datasets_to_combine)
        else:
            combined_dataset = datasets_to_combine[0]
        
        logger.info(f"Combined dataset size: {len(combined_dataset)}")
        
        # Create train/val/test splits
        train, val, test = self.create_data_splits(combined_dataset)
        
        # Tokenize datasets
        train_tokenized = self.tokenize_dataset(train)
        val_tokenized = self.tokenize_dataset(val)
        test_tokenized = self.tokenize_dataset(test)
        
        logger.info("Data preparation completed successfully!")
        
        return train_tokenized, val_tokenized, test_tokenized
    
    def _process_huggingface_dataset(self, dataset, dataset_name: str) -> Optional[Dataset]:
        """
        Process a HuggingFace dataset to match our format.
        
        Args:
            dataset: Raw HuggingFace dataset
            dataset_name: Name of the dataset for specific processing
            
        Returns:
            Processed dataset or None if processing fails
        """
        try:
            if dataset_name == "nvidia/Aegis-AI-Content-Safety-Dataset-2.0":
                return self._process_nvidia_aegis(dataset)
            elif dataset_name == "lmsys/toxic-chat":
                return self._process_toxic_chat(dataset)
            elif dataset_name == "PKU-Alignment/PKU-SafeRLHF-QA":
                return self._process_pku_saferhlf_qa(dataset)
            elif dataset_name == "GuardrailsAI/content-moderation":
                return self._process_guardrails_moderation(dataset)
            elif dataset_name == "ethos":
                return self._process_ethos(dataset)
            elif dataset_name == "toxigen/toxigen-data":
                return self._process_toxigen(dataset)
            elif dataset_name == "google/jigsaw_toxicity_pred":
                return self._process_jigsaw_toxicity(dataset)
            elif dataset_name == "unitary/toxic-bert":
                return self._process_toxic_bert(dataset)
            else:
                logger.warning(f"No specific processor for dataset {dataset_name}, skipping")
                return None
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {e}")
            return None
    
    def _process_nvidia_aegis(self, dataset) -> Dataset:
        """Process the NVIDIA Aegis AI Content Safety Dataset."""
        # Use the train split if available
        if 'train' in dataset:
            data = dataset['train']
        else:
            data = dataset[list(dataset.keys())[0]]
        
        texts = []
        labels = []
        
        # Map Aegis categories to our 4-category system
        aegis_to_ours = {
            'hate': 0,  # hate_speech
            'self_harm': 1,  # self_harm  
            'violence': 2,  # dangerous_advice (broad interpretation)
            'harassment': 3,  # harassment
        }
        
        for example in data:
            # Use prompt as text (could also use response or both)
            text = example.get('prompt', '')
            violated_categories = example.get('violated_categories', '')
            
            if text:
                texts.append(text)
                
                # Create multi-hot label
                label = [0, 0, 0, 0]
                if violated_categories and violated_categories != 'Safe':
                    # Parse comma-separated categories
                    categories = [cat.strip().lower() for cat in violated_categories.split(',')]
                    for category in categories:
                        # Map to our categories (simplified mapping)
                        if any(hate_term in category for hate_term in ['hate', 'identity']):
                            label[0] = 1
                        elif any(harm_term in category for harm_term in ['self', 'harm', 'suicide']):
                            label[1] = 1
                        elif any(violence_term in category for violence_term in ['violence', 'weapon', 'criminal', 'threat']):
                            label[2] = 1
                        elif 'harassment' in category:
                            label[3] = 1
                
                labels.append(label)
        
        return Dataset.from_dict({
            'text': texts,
            'labels': labels,
            'source': ['nvidia-aegis'] * len(texts)
        })
    
    def _process_toxic_chat(self, dataset) -> Dataset:
        """Process the toxic-chat dataset."""
        # Use the train split if available
        if 'train' in dataset:
            data = dataset['train']
        else:
            data = dataset[list(dataset.keys())[0]]
        
        texts = []
        labels = []
        
        for example in data:
            text = example.get('user_input', '')
            toxicity = example.get('toxicity', 0)
            
            if text and toxicity is not None:
                texts.append(text)
                # Convert toxicity to multi-hot label - map to general harassment category
                label = [0, 0, 0, 0]
                if toxicity > 0:
                    label[3] = 1  # harassment category for general toxicity
                labels.append(label)
        
        return Dataset.from_dict({
            'text': texts,
            'labels': labels,
            'source': ['toxic-chat'] * len(texts)
        })
    
    def _process_jigsaw_toxicity(self, dataset) -> Dataset:
        """Process the jigsaw toxicity dataset."""
        # Use the train split if available
        if 'train' in dataset:
            data = dataset['train']
        else:
            data = dataset[list(dataset.keys())[0]]
        
        texts = []
        labels = []
        
        for example in data:
            text = example.get('comment_text', '')
            toxic = example.get('toxic', 0)
            severe_toxic = example.get('severe_toxic', 0)
            obscene = example.get('obscene', 0)
            threat = example.get('threat', 0)
            insult = example.get('insult', 0)
            identity_hate = example.get('identity_hate', 0)
            
            if text:
                texts.append(text)
                # Map Jigsaw categories to our 4-category system
                label = [0, 0, 0, 0]
                if identity_hate:
                    label[0] = 1  # hate_speech
                if threat:
                    label[2] = 1  # dangerous_advice (threats)
                if toxic or severe_toxic or obscene or insult:
                    label[3] = 1  # harassment (general toxicity)
                
                labels.append(label)
        
        return Dataset.from_dict({
            'text': texts,
            'labels': labels,
            'source': ['jigsaw'] * len(texts)
        })
    
    def _process_toxic_bert(self, dataset) -> Dataset:
        """Process the toxic-bert dataset."""
        # Use the train split if available
        if 'train' in dataset:
            data = dataset['train']
        else:
            data = dataset[list(dataset.keys())[0]]
        
        texts = []
        labels = []
        
        for example in data:
            text = example.get('text', '')
            toxic = example.get('toxic', 0)
            
            if text:
                texts.append(text)
                labels.append(int(toxic))
        
        return Dataset.from_dict({
            'text': texts,
            'label': labels,
            'source': ['toxic-bert'] * len(texts)
        })
    
    def _process_pku_saferhlf_qa(self, dataset) -> Dataset:
        """Process the PKU-SafeRLHF-QA dataset."""
        if 'train' in dataset:
            data = dataset['train']
        else:
            data = dataset[list(dataset.keys())[0]]
        
        texts = []
        labels = []
        
        for example in data:
            prompt = example.get('prompt', '')
            response = example.get('response', '')
            harm_category = example.get('harm_category', '')
            severity = example.get('severity_level', 0)
            
            if prompt:
                text = prompt
                if response:
                    text += " " + response
                
                texts.append(text)
                
                # Map PKU harm categories to our 4-category system
                label = [0, 0, 0, 0]
                if harm_category and severity > 0:
                    # Handle case where harm_category might be a list or string
                    if isinstance(harm_category, list):
                        harm_lower = ' '.join(str(cat).lower() for cat in harm_category)
                    else:
                        harm_lower = str(harm_category).lower()
                    
                    if any(term in harm_lower for term in ['hate', 'discrimination', 'bias']):
                        label[0] = 1  # hate_speech
                    elif any(term in harm_lower for term in ['self-harm', 'suicide']):
                        label[1] = 1  # self_harm
                    elif any(term in harm_lower for term in ['violence', 'weapon', 'illegal', 'dangerous']):
                        label[2] = 1  # dangerous_advice
                    else:
                        label[3] = 1  # harassment (general harmful)
                
                labels.append(label)
        
        return Dataset.from_dict({
            'text': texts,
            'labels': labels,
            'source': ['pku-saferhlf-qa'] * len(texts)
        })
    
    def _process_guardrails_moderation(self, dataset) -> Dataset:
        """Process the GuardrailsAI content-moderation dataset."""
        if 'train' in dataset:
            data = dataset['train']
        else:
            data = dataset[list(dataset.keys())[0]]
        
        texts = []
        labels = []
        
        for example in data:
            text = example.get('comment_text', '')
            toxic = example.get('toxic', 0)
            severe_toxic = example.get('severe_toxic', 0) 
            obscene = example.get('obscene', 0)
            threat = example.get('threat', 0)
            insult = example.get('insult', 0)
            identity_hate = example.get('identity_hate', 0)
            
            if text:
                texts.append(text)
                
                # Map to our 4-category system
                label = [0, 0, 0, 0]
                if identity_hate:
                    label[0] = 1  # hate_speech
                if threat:
                    label[2] = 1  # dangerous_advice
                if toxic or severe_toxic or obscene or insult:
                    label[3] = 1  # harassment
                
                labels.append(label)
        
        return Dataset.from_dict({
            'text': texts,
            'labels': labels,
            'source': ['guardrails-moderation'] * len(texts)
        })
    
    def _process_ethos(self, dataset) -> Dataset:
        """Process the ETHOS multilabel dataset."""
        if 'train' in dataset:
            data = dataset['train']
        else:
            data = dataset[list(dataset.keys())[0]]
        
        texts = []
        labels = []
        
        for example in data:
            text = example.get('text', '')
            violence = example.get('violence', 0)
            directed_vs_generalized = example.get('directed_vs_generalized', 0)
            gender = example.get('gender', 0)
            race = example.get('race', 0)
            national_origin = example.get('national_origin', 0)
            disability = example.get('disability', 0)
            religion = example.get('religion', 0)
            sexual_orientation = example.get('sexual_orientation', 0)
            
            if text:
                texts.append(text)
                
                # Map ETHOS categories to our 4-category system  
                label = [0, 0, 0, 0]
                if any([gender, race, national_origin, disability, religion, sexual_orientation]):
                    label[0] = 1  # hate_speech
                if violence:
                    label[2] = 1  # dangerous_advice (violence)
                if directed_vs_generalized:
                    label[3] = 1  # harassment
                
                labels.append(label)
        
        return Dataset.from_dict({
            'text': texts,
            'labels': labels,
            'source': ['ethos'] * len(texts)
        })
    
    def _process_toxigen(self, dataset) -> Dataset:
        """Process the TOXIGEN implicit hate speech dataset.""" 
        if 'train' in dataset:
            data = dataset['train']
        else:
            data = dataset[list(dataset.keys())[0]]
        
        texts = []
        labels = []
        
        for example in data:
            text = example.get('text', '')
            toxicity = example.get('toxicity', 0)
            target_group = example.get('target_group', '')
            
            if text:
                texts.append(text)
                
                # Map toxicity to our categories
                label = [0, 0, 0, 0]
                if toxicity > 0:
                    # TOXIGEN focuses on implicit hate speech
                    label[0] = 1  # hate_speech
                
                labels.append(label)
        
        return Dataset.from_dict({
            'text': texts,
            'labels': labels,
            'source': ['toxigen'] * len(texts)
        })


def create_data_loaders(config_path: str = "configs/base_config.yaml"):
    """
    Convenience function to create data loaders for training.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (train, val, test) datasets
    """
    loader = SafetyDatasetLoader(config_path)
    return loader.load_and_prepare_data()


if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(level=logging.INFO)
    
    try:
        train, val, test = create_data_loaders()
        print(f"Successfully loaded data:")
        print(f"  Train: {len(train)} examples")
        print(f"  Val: {len(val)} examples") 
        print(f"  Test: {len(test)} examples")
        print(f"  Example: {train[0]}")
    except Exception as e:
        print(f"Error loading data: {e}")