import re
from typing import TypeVar, Type
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class BaseAgent:
    def result_to_json(self, result, model_cls: Type[T]) -> T:
        """
        Parses the raw text output from a CrewAI task into a Pydantic model.
        Handles common markdown wrapping like ```json ... ``` and uses regex
        to isolate the JSON body.
        """
        raw_text = getattr(result, "raw", str(result)).strip()
        
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        json_str = raw_text.strip()
        match = re.search(r'(\{.*\})', json_str, re.DOTALL)
        if match:
            json_str = match.group(1)
            
        try:
            return model_cls.model_validate_json(json_str)
        except Exception as primary_e:
            # Try naive repairs for truncated JSON output (e.g. from LLM max_tokens)
            repairs = [
                json_str + '"}',
                json_str + '"}',
                json_str + '}',
                json_str + '"]}',
                json_str + ']}',
            ]
            if not json_str.endswith('"'):
                repairs.insert(0, json_str + '"}')
                repairs.insert(0, json_str + '"]}')

            for r in repairs:
                try:
                    return model_cls.model_validate_json(r)
                except Exception:
                    pass
            raise primary_e
