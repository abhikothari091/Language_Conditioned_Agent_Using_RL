# =============================================================================
# Prompt Templates for LLM Planner
# =============================================================================
"""
Prompt engineering for instruction parsing.

This file contains all prompts used by the LLM planner.

Prompt Engineering Basics:
--------------------------
1. SYSTEM: Sets the AI's role and behavior
2. USER: The actual input (instruction)
3. EXAMPLES: Few-shot demonstrations
4. FORMAT: Expected output structure

Why Structured Prompts?
-----------------------
- LLMs are sensitive to prompt wording
- Consistent formatting → consistent outputs
- Examples dramatically improve accuracy
- JSON output → easy to parse programmatically

MiniGrid-Specific Context:
--------------------------
The LLM needs to know:
- Available actions: navigate_to, pickup, drop, toggle
- Object types: key, ball, box, door
- Colors: red, green, blue, purple, yellow, grey
- Spatial relations: next to, in front of, behind
"""

# =============================================================================
# System Prompt
# =============================================================================
SYSTEM_PROMPT = """You are a robot navigation planner in a grid world.

Your job is to convert natural language instructions into a sequence of subgoals.

Available actions:
- navigate_to(object): Move to be adjacent to the specified object
- pickup(object): Pick up an object (must be adjacent to it)
- drop(): Drop the currently held object
- toggle(object): Open/close a door (must be adjacent to it)

Object format: "<color> <type>" (e.g., "red ball", "blue key", "yellow door")

Rules:
1. To pick something up, you must first navigate to it
2. To open a door, you must first navigate to it
3. Some doors require keys - you must pick up the matching color key first
4. Output ONLY the JSON array, no other text

Always respond with a valid JSON array of subgoal objects."""

# =============================================================================
# Few-Shot Examples
# =============================================================================
FEW_SHOT_EXAMPLES = [
    {
        "instruction": "go to the red ball",
        "plan": [
            {"action": "navigate_to", "target": "red ball"}
        ]
    },
    {
        "instruction": "pick up the blue key",
        "plan": [
            {"action": "navigate_to", "target": "blue key"},
            {"action": "pickup", "target": "blue key"}
        ]
    },
    {
        "instruction": "open the yellow door",
        "plan": [
            {"action": "navigate_to", "target": "yellow door"},
            {"action": "toggle", "target": "yellow door"}
        ]
    },
    {
        "instruction": "put the green box next to the red ball",
        "plan": [
            {"action": "navigate_to", "target": "green box"},
            {"action": "pickup", "target": "green box"},
            {"action": "navigate_to", "target": "red ball"},
            {"action": "drop", "target": None}
        ]
    },
    {
        "instruction": "pick up the blue key and open the blue door",
        "plan": [
            {"action": "navigate_to", "target": "blue key"},
            {"action": "pickup", "target": "blue key"},
            {"action": "navigate_to", "target": "blue door"},
            {"action": "toggle", "target": "blue door"}
        ]
    },
]

# =============================================================================
# Prompt Templates
# =============================================================================
def build_planning_prompt(instruction: str, include_examples: bool = True) -> str:
    """
    Build the complete prompt for the LLM planner.
    
    Parameters:
    -----------
    instruction : str
        The natural language instruction to parse
    include_examples : bool
        Whether to include few-shot examples (recommended)
        
    Returns:
    --------
    str
        The formatted prompt
    """
    prompt_parts = [SYSTEM_PROMPT, ""]
    
    if include_examples:
        prompt_parts.append("Examples:")
        for ex in FEW_SHOT_EXAMPLES:
            prompt_parts.append(f"\nInstruction: {ex['instruction']}")
            import json
            prompt_parts.append(f"Plan: {json.dumps(ex['plan'])}")
        prompt_parts.append("")
    
    prompt_parts.append(f"Instruction: {instruction}")
    prompt_parts.append("Plan:")
    
    return "\n".join(prompt_parts)


def build_chat_messages(instruction: str, include_examples: bool = True) -> list:
    """
    Build chat messages for chat-style models (Llama-3-Instruct).
    
    Modern chat models expect a list of messages with roles.
    This is the preferred format for instruction-tuned models.
    
    Parameters:
    -----------
    instruction : str
        The natural language instruction
    include_examples : bool
        Whether to include few-shot examples
        
    Returns:
    --------
    list
        List of message dicts with 'role' and 'content'
    """
    import json
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    if include_examples:
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": f"Instruction: {ex['instruction']}"
            })
            messages.append({
                "role": "assistant", 
                "content": json.dumps(ex['plan'])
            })
    
    messages.append({
        "role": "user",
        "content": f"Instruction: {instruction}"
    })
    
    return messages


# =============================================================================
# Exported constants
# =============================================================================
PLANNING_PROMPTS = {
    "system": SYSTEM_PROMPT,
    "examples": FEW_SHOT_EXAMPLES,
    "build_prompt": build_planning_prompt,
    "build_chat_messages": build_chat_messages,
}


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    print("Testing prompt templates...")
    print()
    
    # Test text prompt
    text_prompt = build_planning_prompt("go to the purple box")
    print("=== Text Prompt ===")
    print(text_prompt)
    print()
    
    # Test chat messages
    chat_msgs = build_chat_messages("pick up the green key and open the green door")
    print("=== Chat Messages ===")
    for msg in chat_msgs:
        print(f"[{msg['role'].upper()}]: {msg['content'][:80]}...")
    print()
    
    print("✓ Prompt templates test passed!")
