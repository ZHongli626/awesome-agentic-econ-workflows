"""
Model Design Stage - Automated Mode (No Firecrawl, No HITL)
This script formulates formal mathematical models based on theoretical frameworks from Stage 1.

Pipeline:
1. ModelDesigner: Formal mathematical model formulation (complete equation system, constraints, dynamics)

Input: Theoretical frameworks from Stage 1 (theory_output.json)
Output: Formal mathematical model with complete equation system
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# ========== DATA MODELS ==========

class TheoreticalFramework(BaseModel):
    """Model for theoretical framework from Stage 1 (simplified)."""
    framework_title: str
    research_question: str
    theoretical_approach: str
    assumptions: List[Dict]
    conceptual_components: List[Dict]
    mathematical_formulations: List[Dict]
    model_structure: str
    equilibrium_concept: Optional[str] = ""
    solution_approach: str


class ModelVariable(BaseModel):
    """Model for a variable in the mathematical model."""
    variable_id: str = Field(description="Unique variable identifier")
    variable_symbol: str = Field(description="Mathematical symbol (e.g., 'c_t', 'Y')")
    variable_name: str = Field(description="Full name")
    variable_type: str = Field(description="Type: endogenous, exogenous, state, control, parameter")
    description: str = Field(description="Description of the variable")
    domain: str = Field(description="Domain/range (e.g., 'R+', '[0,1]', 'R')")
    time_subscript: bool = Field(description="Whether variable has time subscript")
    agent_subscript: Optional[str] = Field(description="Agent subscript if applicable", default="")


class ModelParameter(BaseModel):
    """Model for a parameter in the mathematical model."""
    parameter_id: str = Field(description="Unique parameter identifier")
    parameter_symbol: str = Field(description="Mathematical symbol (e.g., 'beta', 'sigma')")
    parameter_name: str = Field(description="Full name")
    description: str = Field(description="Economic interpretation")
    typical_range: str = Field(description="Typical range in literature")
    calibration_source: str = Field(description="How to calibrate/estimate")


class ModelEquation(BaseModel):
    """Model for an equation in the mathematical model."""
    equation_id: str = Field(description="Unique equation identifier")
    equation_type: str = Field(description="Type: behavioral, equilibrium, identity, constraint, law_of_motion")
    equation_name: str = Field(description="Name of the equation")
    equation_latex: str = Field(description="LaTeX representation")
    equation_plain: str = Field(description="Plain text representation")
    variables_used: List[str] = Field(description="Variable symbols used")
    parameters_used: List[str] = Field(description="Parameter symbols used")
    interpretation: str = Field(description="Economic interpretation")
    derivation: str = Field(description="How equation is derived")
    timing: Optional[str] = Field(description="Timing convention if applicable", default="")


class ModelConstraint(BaseModel):
    """Model for a constraint in the mathematical model."""
    constraint_id: str = Field(description="Unique constraint identifier")
    constraint_type: str = Field(description="Type: budget, resource, feasibility, non_negativity, boundary")
    constraint_latex: str = Field(description="LaTeX representation")
    constraint_plain: str = Field(description="Plain text representation")
    description: str = Field(description="Description of the constraint")
    binding_conditions: str = Field(description="When constraint binds")


class ModelDynamics(BaseModel):
    """Model for dynamic structure of the model."""
    time_structure: str = Field(description="Discrete/Continuous time")
    time_horizon: str = Field(description="Finite/Infinite horizon")
    state_variables: List[str] = Field(description="State variable symbols")
    control_variables: List[str] = Field(description="Control variable symbols")
    transition_equations: List[str] = Field(description="State transition equation IDs")
    initial_conditions: List[str] = Field(description="Initial conditions")
    terminal_conditions: List[str] = Field(description="Terminal conditions if applicable")


class EquilibriumDefinition(BaseModel):
    """Model for equilibrium definition."""
    equilibrium_type: str = Field(description="Type: competitive, Nash, rational_expectations, etc.")
    equilibrium_conditions: List[str] = Field(description="Conditions defining equilibrium")
    market_clearing: List[str] = Field(description="Market clearing conditions")
    optimality_conditions: List[str] = Field(description="Optimality conditions (FOCs, etc.)")
    consistency_requirements: List[str] = Field(description="Consistency requirements")


class SolutionMethod(BaseModel):
    """Model for solution methodology."""
    solution_approach: str = Field(description="Analytical/Numerical/Simulation")
    solution_steps: List[str] = Field(description="Steps to solve the model")
    computational_methods: List[str] = Field(description="Computational methods needed")
    software_requirements: List[str] = Field(description="Software/packages needed")
    expected_challenges: List[str] = Field(description="Expected computational challenges")


class FormalMathematicalModel(BaseModel):
    """Model for the complete formal mathematical model."""
    model_title: str = Field(description="Title of the mathematical model")
    model_summary: str = Field(description="High-level summary")
    based_on_framework: str = Field(description="Source theoretical framework title")
    
    # Model components
    variables: List[ModelVariable] = Field(description="All variables")
    parameters: List[ModelParameter] = Field(description="All parameters")
    equations: List[ModelEquation] = Field(description="Complete equation system")
    constraints: List[ModelConstraint] = Field(description="All constraints")
    
    # Structure
    dynamics: ModelDynamics = Field(description="Dynamic structure")
    equilibrium: EquilibriumDefinition = Field(description="Equilibrium definition")
    solution_method: SolutionMethod = Field(description="Solution methodology")
    
    # Additional information
    model_notation: str = Field(description="Notation conventions")
    model_assumptions_recap: List[str] = Field(description="Key assumptions recap")
    model_extensions: List[str] = Field(description="Possible extensions")
    
    # Metadata
    metadata: Dict = Field(description="Model metadata", default_factory=dict)


class ModelDesignOutput(BaseModel):
    """Model for the complete model design stage output."""
    theoretical_frameworks: List[TheoreticalFramework] = Field(description="Input frameworks")
    formal_models: List[FormalMathematicalModel] = Field(description="Designed formal models")
    metadata: Dict = Field(description="Output metadata", default_factory=dict)


# ========== AGENT ==========

class ModelDesigner:
    """Agent for formulating formal mathematical models from theoretical frameworks."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "ModelDesigner"
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=self.api_key
        )
    
    def design_formal_model(
        self,
        framework: TheoreticalFramework
    ) -> FormalMathematicalModel:
        """Design a formal mathematical model from a theoretical framework."""
        
        print(f"\n[{self.agent_name}] Designing formal model for:")
        print(f"  {framework.framework_title}")
        
        # Step 1: Define variables
        print(f"  Step 1: Defining model variables...")
        variables = self._define_variables(framework)
        
        # Step 2: Define parameters
        print(f"  Step 2: Defining model parameters...")
        parameters = self._define_parameters(framework)
        
        # Step 3: Formulate complete equation system
        print(f"  Step 3: Formulating complete equation system...")
        equations = self._formulate_equations(framework, variables, parameters)
        
        # Step 4: Specify constraints
        print(f"  Step 4: Specifying constraints...")
        constraints = self._specify_constraints(framework, variables, parameters)
        
        # Step 5: Define dynamics
        print(f"  Step 5: Defining model dynamics...")
        dynamics = self._define_dynamics(framework, variables, equations)
        
        # Step 6: Define equilibrium
        print(f"  Step 6: Defining equilibrium concept...")
        equilibrium = self._define_equilibrium(framework, equations)
        
        # Step 7: Specify solution method
        print(f"  Step 7: Specifying solution method...")
        solution_method = self._specify_solution_method(framework, equations, dynamics)
        
        # Step 8: Synthesize complete model
        print(f"  Step 8: Synthesizing complete formal model...")
        formal_model = self._synthesize_model(
            framework,
            variables,
            parameters,
            equations,
            constraints,
            dynamics,
            equilibrium,
            solution_method
        )
        
        print(f"[{self.agent_name}] Formal model designed: {formal_model.model_title}")
        return formal_model
    
    def _define_variables(
        self,
        framework: TheoreticalFramework
    ) -> List[ModelVariable]:
        """Define all model variables."""
        
        # Prepare context
        components_text = "\n".join([
            f"- {c.get('component_name', 'Unknown')}: {c.get('description', '')[:100]}"
            for c in framework.conceptual_components
        ])
        
        equations_text = "\n".join([
            f"- {e.get('equation_name', 'Unknown')}: {', '.join(e.get('variables', []))}"
            for e in framework.mathematical_formulations
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at defining variables for formal economic models."),
            ("user", """Define 10-20 key variables for this formal mathematical model.

Framework: {framework_title}
Theoretical Approach: {approach}

Components:
{components}

Existing Equations:
{equations}

For each variable, provide:
- variable_id: Short ID (e.g., "V1", "V2")
- variable_symbol: Math symbol (e.g., "c_t", "K", "pi")
- variable_name: Full name
- variable_type: endogenous, exogenous, state, control, or parameter
- description: Clear description
- domain: Domain/range (e.g., "R+", "[0,1]")
- time_subscript: true/false
- agent_subscript: If applicable (e.g., "i", "h")

Return as JSON with "variables" array.
Example: {{
  "variables": [
    {{
      "variable_id": "V1",
      "variable_symbol": "c_t",
      "variable_name": "Consumption",
      "variable_type": "control",
      "description": "Household consumption at time t",
      "domain": "R+",
      "time_subscript": true,
      "agent_subscript": ""
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "framework_title": framework.framework_title,
                "approach": framework.theoretical_approach,
                "components": components_text,
                "equations": equations_text
            })
            
            data = json.loads(result.content)
            variables = [ModelVariable(**v) for v in data.get("variables", [])]
            
            return variables
        
        except Exception as e:
            print(f"    Error defining variables: {e}")
            return []
    
    def _define_parameters(
        self,
        framework: TheoreticalFramework
    ) -> List[ModelParameter]:
        """Define all model parameters."""
        
        equations_text = "\n".join([
            f"- {e.get('equation_name', 'Unknown')}: {', '.join(e.get('parameters', []))}"
            for e in framework.mathematical_formulations
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at defining parameters for formal economic models."),
            ("user", """Define 8-15 key parameters for this formal mathematical model.

Framework: {framework_title}
Theoretical Approach: {approach}

Existing Equations:
{equations}

For each parameter, provide:
- parameter_id: Short ID (e.g., "P1", "P2")
- parameter_symbol: Math symbol (e.g., "beta", "sigma", "alpha")
- parameter_name: Full name
- description: Economic interpretation
- typical_range: Typical range in literature
- calibration_source: How to calibrate/estimate

Return as JSON with "parameters" array.
Example: {{
  "parameters": [
    {{
      "parameter_id": "P1",
      "parameter_symbol": "beta",
      "parameter_name": "Discount factor",
      "description": "Household's discount factor for future utility",
      "typical_range": "[0.95, 0.99]",
      "calibration_source": "Standard macro literature, typically 0.96-0.99"
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "framework_title": framework.framework_title,
                "approach": framework.theoretical_approach,
                "equations": equations_text
            })
            
            data = json.loads(result.content)
            parameters = [ModelParameter(**p) for p in data.get("parameters", [])]
            
            return parameters
        
        except Exception as e:
            print(f"    Error defining parameters: {e}")
            return []
    
    def _formulate_equations(
        self,
        framework: TheoreticalFramework,
        variables: List[ModelVariable],
        parameters: List[ModelParameter]
    ) -> List[ModelEquation]:
        """Formulate complete equation system."""
        
        vars_text = "\n".join([f"- {v.variable_symbol}: {v.variable_name}" for v in variables[:15]])
        params_text = "\n".join([f"- {p.parameter_symbol}: {p.parameter_name}" for p in parameters[:10]])
        
        existing_eqs = "\n".join([
            f"- {e.get('equation_name', 'Unknown')}: {e.get('equation_latex', '')}"
            for e in framework.mathematical_formulations
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at formulating complete equation systems for economic models."),
            ("user", """Formulate a complete equation system (12-20 equations) for this formal model.

Framework: {framework_title}
Approach: {approach}
Model Structure: {structure}

Variables:
{variables}

Parameters:
{parameters}

Existing Equations:
{existing_equations}

For each equation, provide:
- equation_id: Short ID (e.g., "EQ1", "EQ2")
- equation_type: behavioral, equilibrium, identity, constraint, or law_of_motion
- equation_name: Name
- equation_latex: LaTeX (use \\\\frac, \\\\sum, etc.)
- equation_plain: Plain text version
- variables_used: List of variable symbols
- parameters_used: List of parameter symbols
- interpretation: Economic interpretation
- derivation: Brief derivation notes
- timing: Timing convention if relevant

Include: utility/profit maximization FOCs, budget constraints, market clearing, laws of motion, identities.

Return as JSON with "equations" array.
Example: {{
  "equations": [
    {{
      "equation_id": "EQ1",
      "equation_type": "behavioral",
      "equation_name": "Euler Equation",
      "equation_latex": "u'(c_t) = \\\\beta E_t[u'(c_{{t+1}})(1+r_{{t+1}})]",
      "equation_plain": "u'(c_t) = beta * E_t[u'(c_{t+1}) * (1 + r_{t+1})]",
      "variables_used": ["c_t", "c_{t+1}", "r_{t+1}"],
      "parameters_used": ["beta"],
      "interpretation": "Intertemporal consumption optimality",
      "derivation": "FOC from household utility maximization",
      "timing": "Period t decision"
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "framework_title": framework.framework_title,
                "approach": framework.theoretical_approach,
                "structure": framework.model_structure[:300],
                "variables": vars_text,
                "parameters": params_text,
                "existing_equations": existing_eqs
            })
            
            data = json.loads(result.content)
            equations = [ModelEquation(**e) for e in data.get("equations", [])]
            
            return equations
        
        except Exception as e:
            print(f"    Error formulating equations: {e}")
            return []
    
    def _specify_constraints(
        self,
        framework: TheoreticalFramework,
        variables: List[ModelVariable],
        parameters: List[ModelParameter]
    ) -> List[ModelConstraint]:
        """Specify model constraints."""
        
        vars_text = "\n".join([f"- {v.variable_symbol}: {v.variable_name}" for v in variables[:15]])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at specifying constraints for economic models."),
            ("user", """Specify 5-10 key constraints for this formal model.

Framework: {framework_title}

Variables:
{variables}

For each constraint, provide:
- constraint_id: Short ID (e.g., "C1", "C2")
- constraint_type: budget, resource, feasibility, non_negativity, or boundary
- constraint_latex: LaTeX representation
- constraint_plain: Plain text
- description: Description
- binding_conditions: When it binds

Return as JSON with "constraints" array.
Example: {{
  "constraints": [
    {{
      "constraint_id": "C1",
      "constraint_type": "budget",
      "constraint_latex": "c_t + s_t \\\\leq w_t + (1+r_t)s_{{t-1}}",
      "constraint_plain": "c_t + s_t <= w_t + (1+r_t)*s_{t-1}",
      "description": "Household budget constraint",
      "binding_conditions": "Always binding in optimal solution"
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "framework_title": framework.framework_title,
                "variables": vars_text
            })
            
            data = json.loads(result.content)
            constraints = [ModelConstraint(**c) for c in data.get("constraints", [])]
            
            return constraints
        
        except Exception as e:
            print(f"    Error specifying constraints: {e}")
            return []
    
    def _define_dynamics(
        self,
        framework: TheoreticalFramework,
        variables: List[ModelVariable],
        equations: List[ModelEquation]
    ) -> ModelDynamics:
        """Define model dynamics."""
        
        state_vars = [v.variable_symbol for v in variables if v.variable_type == "state"]
        control_vars = [v.variable_symbol for v in variables if v.variable_type == "control"]
        transition_eqs = [e.equation_id for e in equations if e.equation_type == "law_of_motion"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at defining dynamic structures for economic models."),
            ("user", """Define the dynamic structure for this model.

Framework: {framework_title}
Approach: {approach}

State Variables: {state_vars}
Control Variables: {control_vars}
Transition Equations: {transition_eqs}

Provide:
- time_structure: "Discrete" or "Continuous"
- time_horizon: "Finite" or "Infinite"
- state_variables: List of state variable symbols
- control_variables: List of control variable symbols
- transition_equations: List of transition equation IDs
- initial_conditions: List of initial conditions
- terminal_conditions: List of terminal conditions (if finite horizon)

Return as JSON.
Example: {{
  "time_structure": "Discrete",
  "time_horizon": "Infinite",
  "state_variables": ["k_t", "z_t"],
  "control_variables": ["c_t", "l_t"],
  "transition_equations": ["EQ5", "EQ6"],
  "initial_conditions": ["k_0 given", "z_0 ~ N(0, sigma_z^2)"],
  "terminal_conditions": []
}}

Respond with ONLY the JSON object, no other text.
""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "framework_title": framework.framework_title,
                "approach": framework.theoretical_approach,
                "state_vars": ", ".join(state_vars) if state_vars else "None specified",
                "control_vars": ", ".join(control_vars) if control_vars else "None specified",
                "transition_eqs": ", ".join(transition_eqs) if transition_eqs else "None specified"
            })
            
            data = json.loads(result.content)
            dynamics = ModelDynamics(**data)
            
            return dynamics
        
        except Exception as e:
            print(f"    Error defining dynamics: {e}")
            return ModelDynamics(
                time_structure="Discrete",
                time_horizon="Infinite",
                state_variables=state_vars,
                control_variables=control_vars,
                transition_equations=transition_eqs,
                initial_conditions=[],
                terminal_conditions=[]
            )
    
    def _define_equilibrium(
        self,
        framework: TheoreticalFramework,
        equations: List[ModelEquation]
    ) -> EquilibriumDefinition:
        """Define equilibrium concept."""
        
        eq_eqs = [e.equation_id for e in equations if e.equation_type == "equilibrium"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at defining equilibrium concepts for economic models."),
            ("user", """Define the equilibrium concept for this model.

Framework: {framework_title}
Equilibrium Concept: {eq_concept}

Provide:
- equilibrium_type: Type (e.g., "Competitive Equilibrium", "Nash Equilibrium", "Rational Expectations")
- equilibrium_conditions: List of conditions (3-5)
- market_clearing: Market clearing conditions (2-4)
- optimality_conditions: Optimality conditions (3-5)
- consistency_requirements: Consistency requirements (2-3)

Return as JSON.
Example: {{
  "equilibrium_type": "Competitive Equilibrium with Rational Expectations",
  "equilibrium_conditions": ["Households optimize", "Firms optimize", "Markets clear"],
  "market_clearing": ["Labor market: L_d = L_s", "Goods market: Y = C + I"],
  "optimality_conditions": ["Euler equation holds", "Labor supply FOC holds"],
  "consistency_requirements": ["Expectations are rational", "Budget constraints satisfied"]
}}

Respond with ONLY the JSON object, no other text.
""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "framework_title": framework.framework_title,
                "eq_concept": framework.equilibrium_concept or "Not specified"
            })
            
            data = json.loads(result.content)
            equilibrium = EquilibriumDefinition(**data)
            
            return equilibrium
        
        except Exception as e:
            print(f"    Error defining equilibrium: {e}")
            return EquilibriumDefinition(
                equilibrium_type="General Equilibrium",
                equilibrium_conditions=[],
                market_clearing=[],
                optimality_conditions=[],
                consistency_requirements=[]
            )
    
    def _specify_solution_method(
        self,
        framework: TheoreticalFramework,
        equations: List[ModelEquation],
        dynamics: ModelDynamics
    ) -> SolutionMethod:
        """Specify solution methodology."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at specifying solution methods for economic models."),
            ("user", """Specify the solution methodology for this model.

Framework: {framework_title}
Solution Approach: {solution_approach}
Time Structure: {time_structure}
Number of Equations: {num_equations}

Provide:
- solution_approach: "Analytical", "Numerical", or "Simulation"
- solution_steps: Ordered steps to solve (5-8 steps)
- computational_methods: Methods needed (e.g., "Linearization", "Value function iteration")
- software_requirements: Software/packages (e.g., "MATLAB", "Dynare", "Python (NumPy, SciPy)")
- expected_challenges: Computational challenges (2-4)

Return as JSON.
Example: {{
  "solution_approach": "Numerical",
  "solution_steps": ["Step 1: Calibrate parameters", "Step 2: Linearize around steady state", "Step 3: Solve linear system"],
  "computational_methods": ["Log-linearization", "Blanchard-Kahn method"],
  "software_requirements": ["Dynare", "MATLAB"],
  "expected_challenges": ["High dimensionality", "Curse of dimensionality"]
}}

Respond with ONLY the JSON object, no other text.
""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "framework_title": framework.framework_title,
                "solution_approach": framework.solution_approach,
                "time_structure": dynamics.time_structure,
                "num_equations": len(equations)
            })
            
            data = json.loads(result.content)
            solution_method = SolutionMethod(**data)
            
            return solution_method
        
        except Exception as e:
            print(f"    Error specifying solution method: {e}")
            return SolutionMethod(
                solution_approach="Numerical",
                solution_steps=[],
                computational_methods=[],
                software_requirements=[],
                expected_challenges=[]
            )
    
    def _synthesize_model(
        self,
        framework: TheoreticalFramework,
        variables: List[ModelVariable],
        parameters: List[ModelParameter],
        equations: List[ModelEquation],
        constraints: List[ModelConstraint],
        dynamics: ModelDynamics,
        equilibrium: EquilibriumDefinition,
        solution_method: SolutionMethod
    ) -> FormalMathematicalModel:
        """Synthesize complete formal model."""
        
        model_title = f"Formal Model: {framework.framework_title}"
        
        # Generate model summary
        model_summary = f"""
This formal mathematical model operationalizes the theoretical framework '{framework.framework_title}'.
It consists of {len(variables)} variables, {len(parameters)} parameters, {len(equations)} equations, and {len(constraints)} constraints.
The model uses a {dynamics.time_structure.lower()} time structure with {dynamics.time_horizon.lower()} horizon.
Equilibrium is defined as {equilibrium.equilibrium_type}.
        """.strip()
        
        # Notation conventions
        notation = """
Time subscripts: t, t+1, t-1
Agent subscripts: i, h, f (if applicable)
Expectations: E_t[·]
Derivatives: ∂/∂x or x'
        """.strip()
        
        # Recap assumptions
        assumptions_recap = [
            f"{a.get('assumption_id', '')}: {a.get('assumption_statement', '')}"
            for a in framework.assumptions[:5]
        ]
        
        # Possible extensions
        extensions = [
            "Introduce heterogeneity across agents",
            "Add additional shocks or frictions",
            "Extend to open economy setting",
            "Incorporate learning or bounded rationality"
        ]
        
        formal_model = FormalMathematicalModel(
            model_title=model_title,
            model_summary=model_summary,
            based_on_framework=framework.framework_title,
            variables=variables,
            parameters=parameters,
            equations=equations,
            constraints=constraints,
            dynamics=dynamics,
            equilibrium=equilibrium,
            solution_method=solution_method,
            model_notation=notation,
            model_assumptions_recap=assumptions_recap,
            model_extensions=extensions,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "num_variables": len(variables),
                "num_parameters": len(parameters),
                "num_equations": len(equations),
                "num_constraints": len(constraints)
            }
        )
        
        return formal_model


# ========== ORCHESTRATOR ==========

class ModelDesignOrchestrator:
    """Orchestrator for the model design stage."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")
        
        self.model_designer = ModelDesigner(self.api_key)
        self.design_output: Optional[ModelDesignOutput] = None
    
    def run_design_pipeline(
        self,
        theory_output: Dict
    ) -> ModelDesignOutput:
        """Run the complete model design pipeline."""
        
        print(f"\n{'='*70}")
        print(f"MODEL DESIGN PIPELINE")
        print(f"{'='*70}")
        
        # Parse theoretical frameworks
        frameworks_data = theory_output.get('theoretical_frameworks', [])
        frameworks = [TheoreticalFramework(**f) for f in frameworks_data]
        
        print(f"Theoretical Frameworks: {len(frameworks)}")
        print(f"{'='*70}\n")
        
        # Design formal models
        formal_models = []
        for framework in frameworks:
            formal_model = self.model_designer.design_formal_model(framework)
            formal_models.append(formal_model)
        
        # Create output
        self.design_output = ModelDesignOutput(
            theoretical_frameworks=frameworks,
            formal_models=formal_models,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "num_frameworks": len(frameworks),
                "num_models": len(formal_models)
            }
        )
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Formal Models Designed: {len(formal_models)}")
        print(f"{'='*70}\n")
        
        return self.design_output
    
    def save_design_output(self, filename: str = "model_design_output.json"):
        """Save design output to JSON."""
        if not self.design_output:
            print("No design output to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.design_output.model_dump(), f, indent=2)
        
        print(f"[Orchestrator] Design output saved to {filename}")
    
    def save_models_text(self, filename: str = "formal_mathematical_models.txt"):
        """Save formal models in readable text format."""
        if not self.design_output:
            print("No design output to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("FORMAL MATHEMATICAL MODELS\n")
            f.write("="*70 + "\n\n")
            
            for i, model in enumerate(self.design_output.formal_models, 1):
                f.write(f"MODEL {i}: {model.model_title}\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"SUMMARY:\n{model.model_summary}\n\n")
                
                f.write(f"NOTATION:\n{model.model_notation}\n\n")
                
                f.write(f"VARIABLES ({len(model.variables)}):\n")
                for var in model.variables:
                    f.write(f"  {var.variable_symbol}: {var.variable_name} ({var.variable_type})\n")
                    f.write(f"     Domain: {var.domain} | {var.description}\n\n")
                
                f.write(f"PARAMETERS ({len(model.parameters)}):\n")
                for param in model.parameters:
                    f.write(f"  {param.parameter_symbol}: {param.parameter_name}\n")
                    f.write(f"     Range: {param.typical_range} | {param.description}\n\n")
                
                f.write(f"EQUATIONS ({len(model.equations)}):\n")
                for eq in model.equations:
                    f.write(f"  {eq.equation_id}. {eq.equation_name} ({eq.equation_type})\n")
                    f.write(f"     LaTeX: {eq.equation_latex}\n")
                    f.write(f"     Plain: {eq.equation_plain}\n")
                    f.write(f"     Interpretation: {eq.interpretation}\n\n")
                
                f.write(f"CONSTRAINTS ({len(model.constraints)}):\n")
                for const in model.constraints:
                    f.write(f"  {const.constraint_id}. {const.constraint_type}\n")
                    f.write(f"     {const.constraint_latex}\n")
                    f.write(f"     {const.description}\n\n")
                
                f.write(f"DYNAMICS:\n")
                f.write(f"  Time Structure: {model.dynamics.time_structure}\n")
                f.write(f"  Time Horizon: {model.dynamics.time_horizon}\n")
                f.write(f"  State Variables: {', '.join(model.dynamics.state_variables)}\n")
                f.write(f"  Control Variables: {', '.join(model.dynamics.control_variables)}\n\n")
                
                f.write(f"EQUILIBRIUM:\n")
                f.write(f"  Type: {model.equilibrium.equilibrium_type}\n")
                f.write(f"  Conditions:\n")
                for cond in model.equilibrium.equilibrium_conditions:
                    f.write(f"    - {cond}\n")
                f.write("\n")
                
                f.write(f"SOLUTION METHOD:\n")
                f.write(f"  Approach: {model.solution_method.solution_approach}\n")
                f.write(f"  Steps:\n")
                for step in model.solution_method.solution_steps:
                    f.write(f"    {step}\n")
                f.write(f"  Software: {', '.join(model.solution_method.software_requirements)}\n\n")
                
                f.write("-"*70 + "\n\n")
        
        print(f"[Orchestrator] Models saved to {filename}")


def main():
    """Main function for model design stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Load theory output
    theory_file = input("Enter path to theory output JSON (from Stage 1): ").strip()
    
    if not os.path.exists(theory_file):
        print(f"Error: {theory_file} not found!")
        return
    
    # Load data
    with open(theory_file, 'r', encoding='utf-8') as f:
        theory_output = json.load(f)
    
    # Run pipeline
    orchestrator = ModelDesignOrchestrator()
    design_output = orchestrator.run_design_pipeline(theory_output)
    
    # Save outputs
    orchestrator.save_design_output("model_design_output.json")
    orchestrator.save_models_text("formal_mathematical_models.txt")
    
    # Print summary
    print("\n" + "="*70)
    print("MODEL DESIGN SUMMARY")
    print("="*70)
    for i, model in enumerate(design_output.formal_models, 1):
        print(f"\n{i}. {model.model_title}")
        print(f"   Variables: {len(model.variables)}")
        print(f"   Parameters: {len(model.parameters)}")
        print(f"   Equations: {len(model.equations)}")
        print(f"   Constraints: {len(model.constraints)}")
        print(f"   Solution: {model.solution_method.solution_approach}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
