from typing import List, TypeVar, Any

T = TypeVar('T')

def add(l1: List[int], l2: List[int]) -> List[int]:
    return [a + b for a, b in zip(l1, l2)]

def substract(l1: List[int], l2: List[int]) -> List[int]:
    return [a - b for a, b in zip(l1, l2)]

def flatten(l: List[List[T]]) -> List[T]:
    return [item for sublist in l for item in sublist]
