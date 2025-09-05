"""
Knowledge Graph for Kerala Ayurveda
3-tier hierarchical structure: Doshas → Herbs → Products
"""
import networkx as nx
from typing import List, Dict, Optional, Set
from loguru import logger


class AyurvedaKnowledgeGraph:
    """
    Three-tier hierarchical graph:
    Level 1: Core Ayurveda concepts (Doshas, body systems)
    Level 2: Herbs, treatments, therapies
    Level 3: Products, contraindications, specific formulations
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_ayurveda_ontology()
        logger.info(f"Initialized Knowledge Graph with {len(self.graph.nodes())} nodes")
    
    def _build_ayurveda_ontology(self):
        """Build the Ayurveda knowledge graph from Kerala Ayurveda corpus"""
        
        # ===== LEVEL 1: CORE CONCEPTS =====
        
        # Doshas
        self.graph.add_node("Vata", 
            type="dosha",
            properties=["light", "dry", "mobile", "cold"],
            governs=["movement", "communication", "nervous_system", "adaptability"],
            imbalance_signs=["restlessness", "worry", "irregular_digestion", "sleep_disturbance"]
        )
        
        self.graph.add_node("Pitta",
            type="dosha",
            properties=["hot", "sharp", "intense", "focused"],
            governs=["digestion", "metabolism", "transformation", "clarity"],
            imbalance_signs=["irritability", "overheating", "skin_sensitivity", "burnout"]
        )
        
        self.graph.add_node("Kapha",
            type="dosha",
            properties=["heavy", "slow", "cool", "stable"],
            governs=["structure", "lubrication", "nourishment", "stability"],
            imbalance_signs=["sluggishness", "heaviness", "excess_mucus", "lethargy"]
        )
        
        # Body Systems
        self.graph.add_node("nervous_system", type="body_system")
        self.graph.add_node("digestive_system", type="body_system")
        self.graph.add_node("circulatory_system", type="body_system")
        self.graph.add_node("immune_system", type="body_system")
        
        # ===== LEVEL 2: HERBS & TREATMENTS =====
        
        # Ashwagandha
        self.graph.add_node("Ashwagandha",
            type="herb",
            sanskrit_name="Withania somnifera",
            category="adaptogen",
            properties=["warming", "strengthening", "calming"],
            benefits=["stress resilience", "sleep support", "strength", "vitality"],
            contraindications=["thyroid conditions", "pregnancy", "autoimmune disorders"],
            dosage="500mg-1000mg daily"
        )
        
        # Brahmi
        self.graph.add_node("Brahmi",
            type="herb",
            sanskrit_name="Bacopa monnieri",
            category="nervine_tonic",
            properties=["cooling", "soothing", "clarifying"],
            benefits=["cognitive support", "calmness", "scalp_nourishment", "mental_clarity"],
            contraindications=[],
            dosage="300mg-500mg daily"
        )
        
        # Triphala
        self.graph.add_node("Triphala",
            type="herbal_formula",
            components=["Amalaki", "Bibhitaki", "Haritaki"],
            category="digestive_support",
            properties=["balancing", "gentle", "cleansing"],
            benefits=["digestive comfort", "regular elimination", "rejuvenation", "detoxification"],
            contraindications=["chronic digestive disease", "pregnancy", "post-surgery"],
            dosage="1-2 capsules daily"
        )
        
        # Treatments
        self.graph.add_node("Panchakarma",
            type="treatment",
            description="Traditional Ayurvedic detoxification therapy",
            benefits=["deep_cleansing", "dosha_balance", "rejuvenation"],
            contraindications=["pregnancy", "acute_illness", "menstruation"]
        )
        
        self.graph.add_node("Abhyanga",
            type="treatment",
            description="Traditional oil massage therapy",
            benefits=["stress_relief", "circulation", "nervous_system_support"],
            contraindications=["fever", "acute_inflammation", "skin_conditions"]
        )
        
        self.graph.add_node("Shirodhara",
            type="treatment",
            description="Gentle stream of oil over forehead",
            benefits=["deep_relaxation", "stress_relief", "mental_clarity"],
            contraindications=["pregnancy", "recent_head_injury", "fever"]
        )
        
        # ===== LEVEL 3: PRODUCTS =====
        
        # Products from catalog
        self.graph.add_node("KA-P001",
            type="product",
            name="Triphala Capsules",
            category="digestive_support",
            key_herb="Triphala"
        )
        
        self.graph.add_node("KA-P002",
            type="product",
            name="Ashwagandha Stress Balance Tablets",
            category="stress_and_sleep",
            key_herb="Ashwagandha"
        )
        
        self.graph.add_node("KA-P003",
            type="product",
            name="Brahmi Tailam – Head & Hair Oil",
            category="topical_oil",
            key_herb="Brahmi"
        )
        
        # Patient profiles
        self.graph.add_node("elderly", type="patient_profile")
        self.graph.add_node("pregnancy", type="condition")
        self.graph.add_node("thyroid_condition", type="condition")
        self.graph.add_node("autoimmune_condition", type="condition")
        
        # ===== RELATIONSHIPS =====
        
        # Dosha → Body System
        self.graph.add_edge("Vata", "nervous_system", relationship="governs")
        self.graph.add_edge("Pitta", "digestive_system", relationship="governs")
        self.graph.add_edge("Kapha", "immune_system", relationship="governs")
        
        # Herb → Dosha (balancing)
        self.graph.add_edge("Ashwagandha", "Vata", relationship="balances")
        self.graph.add_edge("Ashwagandha", "nervous_system", relationship="supports")
        self.graph.add_edge("Brahmi", "Vata", relationship="calms")
        self.graph.add_edge("Brahmi", "Pitta", relationship="cools")
        self.graph.add_edge("Triphala", "Kapha", relationship="lightens")
        self.graph.add_edge("Triphala", "digestive_system", relationship="supports")
        
        # Product → Herb
        self.graph.add_edge("KA-P001", "Triphala", relationship="contains")
        self.graph.add_edge("KA-P002", "Ashwagandha", relationship="contains")
        self.graph.add_edge("KA-P003", "Brahmi", relationship="contains")
        
        # Contraindications
        self.graph.add_edge("Ashwagandha", "pregnancy", relationship="contraindicated_in")
        self.graph.add_edge("Ashwagandha", "thyroid_condition", relationship="caution_required")
        self.graph.add_edge("Ashwagandha", "autoimmune_condition", relationship="caution_required")
        
        self.graph.add_edge("Triphala", "pregnancy", relationship="contraindicated_in")
        self.graph.add_edge("Triphala", "chronic_digestive_disease", relationship="contraindicated_in")
        
        self.graph.add_edge("Panchakarma", "pregnancy", relationship="contraindicated_in")
        self.graph.add_edge("Abhyanga", "fever", relationship="contraindicated_in")
        
        # Treatment → Dosha
        self.graph.add_edge("Abhyanga", "Vata", relationship="pacifies")
        self.graph.add_edge("Shirodhara", "Vata", relationship="pacifies")
        self.graph.add_edge("Shirodhara", "Pitta", relationship="cools")
    
    def get_related_entities(self, entity: str, radius: int = 2) -> Dict[str, any]:
        """
        Get entities related to given entity within radius
        """
        if entity not in self.graph:
            return {}
        
        # Get ego graph (subgraph centered on entity)
        ego = nx.ego_graph(self.graph, entity, radius=radius)
        
        related = {
            "entity": entity,
            "properties": self.graph.nodes[entity],
            "connections": []
        }
        
        # Get all neighbors
        for node in ego.nodes():
            if node != entity:
                # Get relationship if edge exists
                relationship = None
                if self.graph.has_edge(entity, node):
                    relationship = self.graph[entity][node].get("relationship")
                elif self.graph.has_edge(node, entity):
                    relationship = self.graph[node][entity].get("relationship")
                
                related["connections"].append({
                    "node": node,
                    "properties": self.graph.nodes[node],
                    "relationship": relationship
                })
        
        return related
    
    def check_contraindication(
        self,
        herb_or_treatment: str,
        patient_conditions: List[str]
    ) -> Dict[str, any]:
        """
        Critical safety check: verify contraindications
        
        Returns:
            {
                "safe": bool,
                "contraindications_found": List[str],
                "cautions_found": List[str],
                "severity": "HIGH" | "MEDIUM" | "LOW"
            }
        """
        if herb_or_treatment not in self.graph:
            return {
                "safe": "UNKNOWN",
                "contraindications_found": [],
                "cautions_found": [],
                "severity": "UNKNOWN",
                "reason": f"{herb_or_treatment} not found in knowledge base"
            }
        
        contraindications = []
        cautions = []
        
        # Check each patient condition
        for condition in patient_conditions:
            # Direct contraindication edge
            if self.graph.has_edge(herb_or_treatment, condition):
                rel = self.graph[herb_or_treatment][condition].get("relationship", "")
                
                if "contraindicated" in rel:
                    contraindications.append(condition)
                elif "caution" in rel:
                    cautions.append(condition)
        
        # Determine safety
        if contraindications:
            return {
                "safe": False,
                "contraindications_found": contraindications,
                "cautions_found": cautions,
                "severity": "HIGH",
                "reason": f"{herb_or_treatment} is contraindicated in: {', '.join(contraindications)}"
            }
        elif cautions:
            return {
                "safe": "CAUTION",
                "contraindications_found": [],
                "cautions_found": cautions,
                "severity": "MEDIUM",
                "reason": f"{herb_or_treatment} requires medical supervision for: {', '.join(cautions)}"
            }
        else:
            return {
                "safe": True,
                "contraindications_found": [],
                "cautions_found": [],
                "severity": "LOW",
                "reason": "No contraindications found"
            }
    
    def find_herbs_for_dosha(self, dosha: str) -> List[Dict[str, str]]:
        """Find herbs that balance/support a specific dosha"""
        if dosha not in self.graph:
            return []
        
        herbs = []
        
        # Find herbs with edges to this dosha
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            
            if node_data.get("type") == "herb":
                if self.graph.has_edge(node, dosha):
                    relationship = self.graph[node][dosha].get("relationship")
                    herbs.append({
                        "herb": node,
                        "relationship": relationship,
                        "properties": node_data
                    })
        
        return herbs
    
    def get_product_info(self, product_id: str) -> Optional[Dict]:
        """Get complete information about a product including herb and contraindications"""
        if product_id not in self.graph:
            return None
        
        product_data = dict(self.graph.nodes[product_id])
        
        # Find connected herb
        for neighbor in self.graph.successors(product_id):
            if self.graph.nodes[neighbor].get("type") == "herb":
                herb_data = self.graph.nodes[neighbor]
                product_data["herb_info"] = {
                    "name": neighbor,
                    "benefits": herb_data.get("benefits", []),
                    "contraindications": herb_data.get("contraindications", [])
                }
        
        return product_data
