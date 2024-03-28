import re
import os
import json
from dataclasses import dataclass

import datasets
import pandas as pd

_CITATION = """\
ddd
"""

_DESCRIPTION = """\
ddd
"""

_HOMEPAGE = "ddd"

_LICENSE = 'ddd'

_URL = "https://huggingface.co/datasets/SpeechLLM/SpokenMedicalQA/resolve/main/csv%2Btranscripts%2Bdev%2Bnoanswer.zip"

@dataclass
class CustomConfig(datasets.BuilderConfig):
    name: str = None
    version: datasets.Version = None
    description: str = None
    schema: str = None
    subset_id: str = None

class SpokenMedicalQA(datasets.GeneratorBasedBuilder):
    
    VERSION = datasets.Version("1.0.6")
    
    MMLU_configs = [
        CustomConfig(
            name="MMLU_" + subject,
            version=datasets.Version("1.0.6"),
            description=f"Source schema in the raw MMLU format.",
            schema="MMLU_" + subject,
            subset_id="MMLU_" + subject,
        ) for subject in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]        
    ]
    
    BUILDER_CONFIGS = [
        CustomConfig(
            name="MedMCQA",
            version=VERSION,
            description="Source schema in the raw MedMCQA format.",
            schema="MedMCQA",
            subset_id="MedMCQA",
        ),
        CustomConfig(
            name="MedQA",
            version=VERSION,
            description="Source schema in the raw MedQA format.",
            schema="MedQA",
            subset_id="MedQA",
        ),
    ] + MMLU_configs

    def _info(self):

        features = datasets.Features(
            {   
                "ID": datasets.Value("string"),
                "ID_original_dataset": datasets.Value("string"),
                "duration": datasets.Value("string"),
                "duration_no_answer": datasets.Value("string"),
                "wav": datasets.Value("string"),
                "wav_no_answer": datasets.Value("string"),
                "spk_id": datasets.Value("string"),
                
                "wrd": datasets.Value("string"),
                "small": datasets.Value("string"),
                "medium": datasets.Value("string"),
                "large-v2": datasets.Value("string"),
                
                "class": datasets.Value("string"),
                
                "audio": datasets.features.Audio(sampling_rate=16_000),
                "label": datasets.ClassLabel(names=["A","B","C","D"]),
            }
        )
    
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        data_dir = dl_manager.download_and_extract(_URL)

        return [
            # datasets.SplitGenerator(
            #     name=datasets.Split.TRAIN,
            #     gen_kwargs={
            #         "split": "train",
            #         "config_name": self.config.name,
            #         "path": data_dir,
            #     },
            # ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     gen_kwargs={
            #         "split": "validation",
            #         "config_name": self.config.name,
            #         "path": data_dir,
            #     },
            # ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "config_name": self.config.name,
                    "path": data_dir,
                },
            ),
        ]

    def _generate_examples(self, split, config_name, path):

        if "MMLU" in config_name and split != "test":
            print("\033[91m Not train or validation data is available in MMLU! \033[0m")
            exit(1)

        file_path = os.path.join(path, f"./{config_name}_{split}.csv")
        
        df = pd.read_csv(file_path, sep=",")
    
        for index, d in df.iterrows():
            
            yield d["ID"], {                
                "ID": d["ID"],
                "ID_original_dataset": d["ID_original_dataset"],
                "duration": d["duration"],
                "duration_no_answer": d["duration_no_answer"],
                "wav": d["wav"],
                "wav_no_answer": d["wav_no_answer"],
                "spk_id": d["spk_id"],
                
                "wrd": d["wrd"],
                "small": d["small"],
                "medium": d["medium"],
                "large-v2": d["large-v2"],
                
                "class": d["class"],
                "label": d["class"],
                
                "audio": d["wav_no_answer"],
            }
            
