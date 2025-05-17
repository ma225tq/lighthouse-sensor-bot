# type: ignore
import typing as t
import os
from dataclasses import dataclass, field
import json
import aiohttp
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import SingleTurnMetric, MetricType

@dataclass
class LenientFactualCorrectness(SingleTurnMetric):
    name: str = "lenient_factual_correctness"
    api_key: str = None
    # Class-level dictionary to store extracted_true_values by reference text
    _extracted_true_values_dict = {}  # Class-level dictionary

    required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"response", "reference", "user_input"}
        }
    )

    def init(self, run_config=None) -> None:
        """Initialize the API key for OpenRouter."""
        # Get API key from environment
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        # Don't clear the class-level dictionary - keep values across instances
        # Instead, optionally clear it at the start of a full test run

    def supports_sample_type(self, sample_type) -> bool:
        return sample_type is SingleTurnSample
    
    # Add a method to register extracted true values
    def register_extracted_true_value(self, reference: str, value: float) -> None:
        """Register an extracted true value for a reference text"""
        if reference and value is not None:
            # Normalize the reference text to avoid matching issues
            normalized_ref = reference.strip()
            LenientFactualCorrectness._extracted_true_values_dict[normalized_ref] = value
            print(f"DEBUG: Registered value {value} for normalized reference: '{normalized_ref}'")

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks=None) -> float:
        # Get the query from either question or user_input attributes
        query = getattr(sample, "question", "") or getattr(sample, "user_input", "") or ""
        
        print("_single_turn_ascore called")
        print(f"DEBUG: Reference text: '{sample.reference}'")
        print(f"DEBUG: Available keys in _extracted_true_values_dict: {list(LenientFactualCorrectness._extracted_true_values_dict.keys())}")
        
        # Try to get the pre-registered extracted true value for this reference
        reference_val = None
        if sample.reference in LenientFactualCorrectness._extracted_true_values_dict:
            reference_val = LenientFactualCorrectness._extracted_true_values_dict[sample.reference]
            print(f"Using pre-registered extracted value: {reference_val}")
        else:
            print(f"DEBUG: No pre-registered value found for this reference")
            # Try a more flexible matching approach
            for ref_text, value in LenientFactualCorrectness._extracted_true_values_dict.items():
                if ref_text.strip() == sample.reference.strip():
                    reference_val = value
                    print(f"DEBUG: Found match using stripped comparison: {reference_val}")
                    break
        
        # Extract relevant number from response based on the query
        response_val = await self.extract_first_number(sample.response, query)
        
        # If we don't have both values, return 0
        if response_val is None or reference_val is None:
            print(f"DEBUG: Missing values - response_val: {response_val}, reference_val: {reference_val}")
            return 0.0
            
        # Compare the numbers
        score = self.compare_numbers(response_val, reference_val)
        print(f"DEBUG: Final score: {score} (comparing {response_val} to {reference_val})")
        return score

    async def extract_first_number(self, text: str, query: str) -> t.Optional[float]:
        """Extract the first relevant number from text using OpenRouter API directly."""
        print(f"DEBUG: response: {text}")
        if not text:
            return None
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost"
        }
        
        payload = {
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a precise numerical extractor. 
                    Given a query and a text, extract the MOST RELEVANT numerical value from the text that answers the query.
                    
                    Output format rules:
                    - Output ONLY a valid decimal number (e.g., 254186.70)
                    - Do NOT use scientific notation
                    - Do NOT include any text, units, or symbols
                    - Do NOT include formatting like asterisks, markdown, or quotes
                    - If no relevant numbers are found, respond with 'None'
                    
                    For example:
                    "The cost was 90.70 SEK" → 90.70
                    "About 5 million dollars" → 5000000
                    "No numbers here" → None"""
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nText: {text}\n\nExtract the most relevant number that answers this query:"
                }
            ],
            "temperature": 0.0
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("https://openrouter.ai/api/v1/chat/completions", 
                                      headers=headers, json=payload) as response:
                    if response.status != 200:
                        print(f"Error from OpenRouter: {await response.text()}")
                        return None
                    
                    data = await response.json()
                    extracted = data["choices"][0]["message"]["content"].strip()
                    
                    # Handle "None" response
                    if extracted.lower() == "none":
                        return None
                        
                    # Clean up the extracted text - strip any markdown or formatting
                    import re
                    # Remove all non-numeric characters except decimal point
                    clean_extracted = re.sub(r'[^\d.]', '', extracted)
                    
                    print(f"Extracted number: '{extracted}' -> Cleaned: '{clean_extracted}'")
                    
                    # Try to convert to float
                    try:
                        return float(clean_extracted)
                    except ValueError:
                        print(f"Failed to extract number from: '{extracted}' -> '{clean_extracted}'")
                        return None
                    
        except Exception as e:
            print(f"Error extracting number: {e}")
            return None

    def compare_numbers(self, num1: float, num2: float) -> float:
        """
        Compare two numbers and return a similarity score between 0.0 and 1.0.
        
        A score of 1.0 means exact match (0% difference)
        A score of 0.5 means 50% difference
        A score of 0.0 means 100% difference or greater
        """
        # If exact match, return perfect score
        if num1 == num2:
            return 1.0
        
        # Calculate the relative error
        relative_error = abs(num1 - num2) / max(abs(num2), 1e-10)
        
        # Simple linear scale: 
        # 0% difference → 1.0
        # 100% difference → 0.0
        
        return max(0.0, 1.0 - relative_error)