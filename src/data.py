"""
Data loading utilities for evaluation datasets.

Provides:
- MATH-500 dataset loader
- Generic dataset interface placeholder
- Validation split handling
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class MathProblem:
    """A math problem with ground truth answer."""
    idx: int
    problem: str
    answer: str
    level: Optional[str] = None
    type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetLoader:
    """Abstract base class for dataset loaders."""
    
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
    
    def __len__(self) -> int:
        raise NotImplementedError
    
    def __iter__(self) -> Iterator[MathProblem]:
        raise NotImplementedError
    
    def __getitem__(self, idx: int) -> MathProblem:
        raise NotImplementedError


class Math500Loader(DatasetLoader):
    """
    Loader for MATH-500 benchmark dataset.
    
    Expected format: JSON array of objects with:
    - "problem": str
    - "answer": str
    - Optional: "level", "type"
    """
    
    def __init__(
        self,
        path: Union[str, Path],
        num_samples: Optional[int] = None,
    ):
        super().__init__(path)
        self.num_samples = num_samples
        self._data: Optional[List[Dict[str, Any]]] = None
    
    def _load_data(self):
        """Load data from JSON file."""
        if self._data is not None:
            return
        
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        
        logger.info(f"Loading MATH-500 from: {self.path}")
        
        with open(self.path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        
        # Limit samples if specified
        if self.num_samples is not None:
            self._data = self._data[:min(self.num_samples, len(self._data))]
        
        logger.info(f"Loaded {len(self._data)} samples")
    
    @property
    def data(self) -> List[Dict[str, Any]]:
        """Lazy-loaded data."""
        if self._data is None:
            self._load_data()
        return self._data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __iter__(self) -> Iterator[MathProblem]:
        for idx, item in enumerate(self.data):
            yield self._to_problem(idx, item)
    
    def __getitem__(self, idx: int) -> MathProblem:
        return self._to_problem(idx, self.data[idx])
    
    def _to_problem(self, idx: int, item: Dict[str, Any]) -> MathProblem:
        """Convert raw item to MathProblem."""
        return MathProblem(
            idx=idx,
            problem=item["problem"],
            answer=item["answer"],
            level=item.get("level"),
            type=item.get("type"),
            metadata={k: v for k, v in item.items() if k not in ["problem", "answer", "level", "type"]},
        )


class ValidationDataLoader(DatasetLoader):
    """
    Placeholder loader for validation data.
    
    TODO: Implement based on actual validation data format.
    """
    
    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        num_samples: Optional[int] = None,
        split: str = "validation",
    ):
        super().__init__(path or "")
        self.num_samples = num_samples
        self.split = split
        self._data: List[Dict[str, Any]] = []
    
    def _load_data(self):
        """
        Load validation data.
        
        TODO: Implement actual loading logic.
        """
        logger.warning("ValidationDataLoader is a placeholder. Implement actual loading.")
        
        # Placeholder: try to load from path if it exists
        if self.path and Path(self.path).exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            
            if self.num_samples:
                self._data = self._data[:self.num_samples]
    
    def __len__(self) -> int:
        if not self._data:
            self._load_data()
        return len(self._data)
    
    def __iter__(self) -> Iterator[MathProblem]:
        if not self._data:
            self._load_data()
        
        for idx, item in enumerate(self._data):
            yield MathProblem(
                idx=idx,
                problem=item.get("problem", ""),
                answer=item.get("answer", ""),
            )
    
    def __getitem__(self, idx: int) -> MathProblem:
        if not self._data:
            self._load_data()
        
        item = self._data[idx]
        return MathProblem(
            idx=idx,
            problem=item.get("problem", ""),
            answer=item.get("answer", ""),
        )


def load_dataset(
    dataset_type: str,
    path: Union[str, Path],
    **kwargs
) -> DatasetLoader:
    """
    Factory function to load datasets.
    
    Args:
        dataset_type: Type of dataset ("math500", "validation", etc.)
        path: Path to dataset file
        **kwargs: Additional arguments for loader
    
    Returns:
        DatasetLoader instance
    """
    if dataset_type == "math500":
        return Math500Loader(path, **kwargs)
    elif dataset_type == "validation":
        return ValidationDataLoader(path, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
