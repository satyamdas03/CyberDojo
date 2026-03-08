"""
CyberDojo Infinite Scenario Generation

Uses LLMs to dynamically generate novel, themed network layouts 
(e.g., Hospital, Power Grid, Bank, Military Base) to replace the 
hardcoded random subnets.
"""

import os
import random
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


class NodeDefinition(BaseModel):
    name: str = Field(description="Name of the node (e.g., 'mri-scanner', 'atm-controller', 'web-portal').")
    subnet_index: int = Field(description="Index of the subnet this node belongs to.")
    os_type: str = Field(description="Operating system: 'linux', 'windows', or 'router'.")
    is_critical: bool = Field(description="Whether this is a high-value target (e.g., main database, central controller). Only 1-2 nodes should be critical.")
    data_value: float = Field(description="Value of data on this node from 1.0 to 10.0.")
    services: List[str] = Field(description="List of services running on this node. Must be from: 'http', 'https', 'ssh', 'ftp', 'smb', 'dns', 'smtp', 'mysql', 'rdp', 'telnet'.")


class SubnetDefinition(BaseModel):
    name: str = Field(description="Name of the subnet (e.g., 'ICU-Ward', 'Corporate-LAN', 'DMZ').")
    is_dmz: bool = Field(description="True if this subnet is public-facing/DMZ.")


class NetworkScenarioPlan(BaseModel):
    theme_name: str = Field(description="A short, cool name for the scenario (e.g., 'Operation: Midnight Bank').")
    subnets: List[SubnetDefinition] = Field(description="The subnets in this scenario. Must have at least 3.")
    nodes: List[NodeDefinition] = Field(description="The nodes scattered across the subnets. Must have 8 to 15 nodes total.")


def generate_scenario(theme: str) -> Optional[Dict]:
    """Generates a dynamic network scenario based on a text theme."""
    if not HAS_LANGCHAIN:
        logger.error("LangChain is required for infinite scenario generation.")
        return None
        
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is missing. Cannot generate scenario.")
        return None

    logger.info(f"Generating infinite scenario with theme: '{theme}'...")
    
    llm = ChatOpenAI(model="gpt-4o", api_key=api_key).with_structured_output(NetworkScenarioPlan)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert cybersecurity range architect. Your job is to generate a highly realistic, "
                   "themed mock network topology for a hacker wargame.\n"
                   "RULES:\n"
                   "1. Subnet 0 MUST be the DMZ (is_dmz=True).\n"
                   "2. Total subnets should be between 3 and 5.\n"
                   "3. Total nodes should be between 8 and 15.\n"
                   "4. Use creative, realistic names based on the theme (e.g. for a hospital: 'xray-machine', 'patient-records-db').\n"
                   "5. Assign realistic services based on the node's purpose.\n"
                   "6. Exactly 1 or 2 nodes must be marked `is_critical=True` as the ultimate crown jewels.\n"),
        ("user", "Design a network scenario with the following theme: {theme}")
    ])
    
    chain = prompt | llm
    
    try:
        result: NetworkScenarioPlan = chain.invoke({"theme": theme})
        
        # Convert Pydantic to a raw dictionary that NetworkConfig can store easily
        scenario_data = {
            "theme_name": result.theme_name,
            "subnets": [{"name": s.name, "is_dmz": s.is_dmz} for s in result.subnets],
            "nodes": []
        }
        
        for n in result.nodes:
            # Safety check: ensure subnet index is valid
            idx = min(n.subnet_index, len(result.subnets) - 1)
            scenario_data["nodes"].append({
                "name": n.name,
                "subnet_index": idx,
                "os_type": n.os_type,
                "is_critical": n.is_critical,
                "data_value": n.data_value,
                "services": n.services
            })
            
        return scenario_data
        
    except Exception as e:
        logger.error(f"Failed to generate scenario: {e}")
        return None
