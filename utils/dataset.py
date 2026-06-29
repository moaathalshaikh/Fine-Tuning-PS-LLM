"""
Dataset utilities for Psychological Safety fine-tuning.
"""

from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.inference import build_prompt


# ==========================================================
# Load Raw Data
# ==========================================================

def load_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw datasets.
    """

    raw_dir = project_root / "data" / "raw"

    quotes = pd.read_csv(
        raw_dir / "quotes.csv"
    )

    gold = pd.read_csv(
        raw_dir / "gold_standard.csv"
    )

    return quotes, gold


# ==========================================================
# Merge Data
# ==========================================================

def merge_data(
    quotes: pd.DataFrame,
    gold: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge quotes with gold labels.
    """

    dataset = quotes.merge(
        gold,
        on="quote_id",
        how="inner",
    )

    dataset = dataset.rename(
        columns={
            "category": "gold_category",
        }
    )

    dataset = dataset[
        [
            "quote_id",
            "quote_text",
            "gold_category",
        ]
    ]

    return dataset


# ==========================================================
# Split Dataset
# ==========================================================

def split_data(
    dataset: pd.DataFrame,
    train_size: float = 0.80,
    validation_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train / validation / test.

    Random split is used because the dataset is small and
    some classes contain very few samples.
    """

    train_df, temp_df = train_test_split(
        dataset,
        train_size=train_size,
        shuffle=True,
        random_state=random_state,
    )

    validation_ratio = validation_size / (1 - train_size)

    validation_df, test_df = train_test_split(
        temp_df,
        train_size=validation_ratio,
        shuffle=True,
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ==========================================================
# Save Dataset Splits
# ==========================================================

def save_splits(
    project_root: Path,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Save train / validation / test CSV files.
    """

    split_dir = project_root / "data" / "splits"

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        split_dir / "train.csv",
        index=False,
    )

    validation_df.to_csv(
        split_dir / "validation.csv",
        index=False,
    )

    test_df.to_csv(
        split_dir / "test.csv",
        index=False,
    )


# ==========================================================
# Build SFT Dataset
# ==========================================================

def build_sft_dataset(
    dataframe: pd.DataFrame,
    prompt_template: str,
) -> list[dict]:
    """
    Build a chat-format dataset for supervised fine-tuning.
    """

    sft_dataset = []

    for _, row in dataframe.iterrows():

        prompt = build_prompt(
            prompt_template=prompt_template,
            quote_id=row["quote_id"],
            quote_text=row["quote_text"],
        )

        assistant = [
            {
                "id_quote": row["quote_id"],
                "category": row["gold_category"],
                "related_quote": row["quote_text"],
            }
        ]

        sft_dataset.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            assistant,
                            ensure_ascii=False,
                        ),
                    },
                ]
            }
        )

    return sft_dataset


# ==========================================================
# Save JSONL
# ==========================================================

def save_jsonl(
    dataset: list[dict],
    output_path: Path,
) -> None:
    """
    Save a list of dictionaries as JSONL.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        for item in dataset:

            json.dump(
                item,
                f,
                ensure_ascii=False,
            )

            f.write("\n")


# ==========================================================
# Build DPO Dataset
# ==========================================================

def build_dpo_dataset():
    """
    Placeholder for DPO dataset generation.
    """

    raise NotImplementedError(
        "build_dpo_dataset() will be implemented in the DPO stage."
    )