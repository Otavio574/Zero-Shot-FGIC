# src/generate_prompts.py
from typing import List, Dict, Optional

DEFAULT_TEMPLATES = [
    "a photo of a {}",
    "a high quality photograph of the {}",
    "a close-up photo of the {}",
    "a detailed image of a {}",
]

def make_prompts_for_class(class_name: str, templates: Optional[List[str]] = None) -> List[str]:
    if templates is None:
        templates = DEFAULT_TEMPLATES
    readable = class_name.replace("_", " ")
    return [t.format(readable) for t in templates]

def make_prompts_for_all(class_names: List[str], templates: Optional[List[str]] = None) -> Dict[str, List[str]]:
    return {c: make_prompts_for_class(c, templates) for c in class_names}
