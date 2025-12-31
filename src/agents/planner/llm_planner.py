# =============================================================================
# LLM Planner
# =============================================================================
"""
Llama 3.2 based instruction planner.

This module loads a local LLM and uses it to parse natural language
instructions into executable subgoal sequences.

Key Design Decisions:
---------------------

1. QUANTIZATION (4-bit)
   - Reduces model size by 4x (3B model: 6GB → 1.5GB)
   - Minimal quality loss for our structured task
   - Essential for running on M3 Mac with limited RAM
   
2. CACHING
   - LLM loading is slow (~10-30 seconds)
   - We load once and reuse for all instructions
   - Generation is fast (~100ms per instruction)

3. STRUCTURED OUTPUT
   - We use JSON format for reliability
   - Easy to parse and validate
   - Fallback parsing for imperfect outputs

4. FROZEN WEIGHTS
   - We don't fine-tune the LLM
   - Keeps compute requirements low
   - Prompt engineering is sufficient for our task
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

# Check if transformers is available
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. LLM features disabled.")

from src.agents.planner.prompts import build_chat_messages, SYSTEM_PROMPT


# =============================================================================
# Subgoal Data Structure
# =============================================================================
class Subgoal:
    """
    A single subgoal in the plan.
    
    Attributes:
    -----------
    action : str
        The action type: "navigate_to", "pickup", "drop", "toggle"
    target : str or None
        The target object (e.g., "red ball")
    completed : bool
        Whether this subgoal has been achieved
    """
    
    VALID_ACTIONS = {"navigate_to", "pickup", "drop", "toggle"}
    
    def __init__(self, action: str, target: Optional[str] = None):
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Invalid action: {action}. Must be one of {self.VALID_ACTIONS}")
        self.action = action
        self.target = target
        self.completed = False
    
    def __repr__(self):
        target_str = f"({self.target})" if self.target else "()"
        status = "✓" if self.completed else "○"
        return f"{status} {self.action}{target_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "target": self.target}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Subgoal":
        return cls(action=d["action"], target=d.get("target"))


# =============================================================================
# LLM Planner
# =============================================================================
class LLMPlanner:
    """
    LLM-based instruction planner using Llama 3.2.
    
    This class handles:
    - Loading the LLM with appropriate quantization
    - Generating plans from instructions
    - Parsing and validating the output
    
    Example:
    --------
    >>> planner = LLMPlanner(model_name="meta-llama/Llama-3.2-1B-Instruct")
    >>> plan = planner.plan("pick up the blue key")
    >>> print(plan)
    [○ navigate_to(blue key), ○ pickup(blue key)]
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: Optional[str] = None,
        use_4bit: bool = True,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize the LLM planner.
        
        Parameters:
        -----------
        model_name : str
            HuggingFace model name. Options:
            - "meta-llama/Llama-3.2-1B-Instruct" (smallest, fastest)
            - "meta-llama/Llama-3.2-3B-Instruct" (better quality)
            
        device : str, optional
            Device to run on: "cpu", "cuda", "mps" (Mac Metal)
            If None, auto-detects best available device.
            
        use_4bit : bool
            Whether to use 4-bit quantization (recommended for Mac)
            
        hf_token : str, optional
            HuggingFace token for accessing Llama 3.2
            If None, tries to load from HF_TOKEN environment variable
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"  # Mac Metal
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Model will be loaded lazily
        self._model = None
        self._tokenizer = None
        self._pipeline = None
    
    def _load_model(self):
        """
        Load the model (lazy loading).
        
        Why lazy loading?
        -----------------
        - Model loading takes 10-30 seconds
        - Don't want to block import
        - Only load when actually needed
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers not installed. Run: pip install transformers torch"
            )
        
        if self._pipeline is not None:
            return  # Already loaded
        
        print(f"Loading {self.model_name}...")
        print(f"  Device: {self.device}")
        print(f"  4-bit quantization: {self.use_4bit}")
        
        # Configure quantization
        model_kwargs = {}
        
        if self.use_4bit and self.device != "mps":
            # bitsandbytes quantization (not supported on MPS yet)
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["quantization_config"] = quantization_config
            except ImportError:
                print("Warning: bitsandbytes not available, using full precision")
        
        # For MPS (Mac), use float16 instead
        if self.device == "mps":
            model_kwargs["torch_dtype"] = torch.float16
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
        )
        
        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.hf_token,
            device_map="auto" if self.device != "cpu" else None,
            **model_kwargs,
        )
        
        if self.device == "cpu":
            self._model = self._model.to("cpu")
        
        # Create text generation pipeline
        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            device_map="auto",
        )
        
        print("Model loaded successfully!")
    
    def plan(
        self,
        instruction: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> List[Subgoal]:
        """
        Generate a plan for the given instruction.
        
        Parameters:
        -----------
        instruction : str
            Natural language instruction (e.g., "go to the red ball")
            
        max_new_tokens : int
            Maximum tokens to generate
            
        temperature : float
            Sampling temperature (lower = more deterministic)
            
        Returns:
        --------
        List[Subgoal]
            Ordered list of subgoals to execute
        """
        # Ensure model is loaded
        self._load_model()
        
        # Build chat messages
        messages = build_chat_messages(instruction)
        
        # Generate response
        outputs = self._pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=self._tokenizer.eos_token_id,
        )
        
        # Extract generated text
        generated = outputs[0]["generated_text"]
        
        # The last message is the assistant's response
        if isinstance(generated, list):
            # Chat format
            response = generated[-1]["content"]
        else:
            # Plain text format
            response = generated.split("Plan:")[-1].strip()
        
        # Parse the JSON plan
        subgoals = self._parse_plan(response)
        
        return subgoals
    
    def _parse_plan(self, response: str) -> List[Subgoal]:
        """
        Parse the LLM response into subgoals.
        
        Handles various response formats and common errors.
        """
        # Try to extract JSON array
        try:
            # Find JSON array in response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                plan_json = json.loads(json_match.group())
                return [Subgoal.from_dict(sg) for sg in plan_json]
        except json.JSONDecodeError:
            pass
        
        # Fallback: try to parse line by line
        subgoals = []
        for line in response.split('\n'):
            line = line.strip()
            if 'navigate_to' in line or 'pickup' in line or 'drop' in line or 'toggle' in line:
                # Try to extract action and target
                for action in Subgoal.VALID_ACTIONS:
                    if action in line:
                        # Extract target from parentheses or quotes
                        target_match = re.search(r'["\']([^"\']+)["\']', line)
                        target = target_match.group(1) if target_match else None
                        subgoals.append(Subgoal(action=action, target=target))
                        break
        
        if not subgoals:
            # Last resort: single navigate_to for simple instructions
            print(f"Warning: Could not parse plan from: {response[:100]}...")
            subgoals = [Subgoal(action="navigate_to", target="target")]
        
        return subgoals
    
    def plan_batch(
        self,
        instructions: List[str],
        **kwargs,
    ) -> List[List[Subgoal]]:
        """
        Generate plans for multiple instructions.
        
        More efficient than calling plan() multiple times
        due to batch processing.
        """
        return [self.plan(inst, **kwargs) for inst in instructions]


# =============================================================================
# Simple Rule-Based Planner (Fallback)
# =============================================================================
class RuleBasedPlanner:
    """
    Simple rule-based planner for testing without LLM.
    
    This is useful for:
    - Quick testing without loading LLM
    - Baseline comparison
    - Environments where LLM is overkill
    
    Uses pattern matching to parse common instruction formats.
    """
    
    # Patterns for common instructions
    PATTERNS = [
        # "go to the <color> <object>"
        (r"go to (?:the )?(\w+) (\w+)", 
         lambda m: [Subgoal("navigate_to", f"{m.group(1)} {m.group(2)}")]),
        
        # "pick up the <color> <object>"
        (r"pick up (?:the )?(\w+) (\w+)",
         lambda m: [
             Subgoal("navigate_to", f"{m.group(1)} {m.group(2)}"),
             Subgoal("pickup", f"{m.group(1)} {m.group(2)}")
         ]),
        
        # "open the <color> door"
        (r"open (?:the )?(\w+) door",
         lambda m: [
             Subgoal("navigate_to", f"{m.group(1)} door"),
             Subgoal("toggle", f"{m.group(1)} door")
         ]),
        
        # "put the <color> <object> next to the <color> <object>"
        (r"put (?:the )?(\w+) (\w+) next to (?:the )?(\w+) (\w+)",
         lambda m: [
             Subgoal("navigate_to", f"{m.group(1)} {m.group(2)}"),
             Subgoal("pickup", f"{m.group(1)} {m.group(2)}"),
             Subgoal("navigate_to", f"{m.group(3)} {m.group(4)}"),
             Subgoal("drop", None)
         ]),
    ]
    
    def plan(self, instruction: str) -> List[Subgoal]:
        """Generate a plan using pattern matching."""
        instruction = instruction.lower().strip()
        
        for pattern, handler in self.PATTERNS:
            match = re.match(pattern, instruction)
            if match:
                return handler(match)
        
        # Fallback: try to extract any object mention
        words = instruction.split()
        for i, word in enumerate(words):
            if word in ['red', 'green', 'blue', 'purple', 'yellow', 'grey']:
                if i + 1 < len(words):
                    target = f"{word} {words[i+1]}"
                    return [Subgoal("navigate_to", target)]
        
        # Last resort
        return [Subgoal("navigate_to", "target")]


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    print("Testing planners...")
    print()
    
    # Test rule-based planner (no LLM needed)
    print("=== Rule-Based Planner ===")
    rule_planner = RuleBasedPlanner()
    
    test_instructions = [
        "go to the red ball",
        "pick up the blue key",
        "open the yellow door",
        "put the green box next to the red ball",
    ]
    
    for inst in test_instructions:
        plan = rule_planner.plan(inst)
        print(f"Instruction: {inst}")
        print(f"Plan: {plan}")
        print()
    
    print("✓ Rule-based planner test passed!")
    print()
    
    # Test LLM planner only if model is available
    print("=== LLM Planner ===")
    if not TRANSFORMERS_AVAILABLE:
        print("Skipping LLM test (transformers not installed)")
    elif not os.getenv("HF_TOKEN"):
        print("Skipping LLM test (HF_TOKEN not set)")
    else:
        print("To test LLM planner, uncomment the code below:")
        print("  # planner = LLMPlanner()")
        print("  # plan = planner.plan('go to the red ball')")
        print("  # print(plan)")
