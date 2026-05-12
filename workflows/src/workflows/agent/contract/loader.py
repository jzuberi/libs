import yaml
from pathlib import Path

def load_agent_contract():
    
    contract_path = Path(__file__).parent / "intents.yaml"
    
    with open(contract_path, "r") as f:
        return yaml.safe_load(f)
