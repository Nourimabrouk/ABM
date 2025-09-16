Memo for the Coding Agent: Technical and Theoretical Specifications for the ABM Project
To: Coding Agent
From: Jan-Willem Simons
Date: 15 September 2025
Subject: Implementation of the SAOM for the Social Norm Intervention Paper

This document provides detailed instructions to guide the coding of the agent-based model (ABM) described in my research proposal. The goal is to translate the micro-theory and experimental design into a robust and flexible simulation framework. Please adhere to the following specifications to ensure the code accurately reflects the intentions and constraints of the research.

1. Core Framework and Technology

Modeling Approach: The entire project must be implemented as a Stochastic Actor-Oriented Model (SAOM). This framework is chosen for its ability to model the co-evolution of social networks and individual behaviors/attitudes through micro-level mechanisms.




Software: All modeling and simulation will be conducted in R, using the RSiena package. A strong proficiency in this package is essential.


Custom Development: As detailed below, this project requires the creation of new, custom effects within the RSiena C++ architecture. The final codebase must include the source code for these custom effects and instructions for their compilation.



2. Data Structure and Preparation
The code must be able to ingest and process the data from the Shani et al. (2023) follow-up study.



Input Data: The raw data consists of 2,585 students nested in 105 classes across 3 waves.


Dependent Variables: The model will simulate the co-evolution of three variables:

Friendship Network: A directed network representing friendship nominations.


Cooperation Network: A directed network representing cooperation on schoolwork.


Tolerance Attitude: An individual-level behavioral variable representing tolerance.

Covariates: The following actor-level attributes must be included:


Ethnicity: A binary variable (majority/minority) to test for interethnic dynamics.




Gender: A binary variable to control for gender homophily.


Prejudice: An actor attribute to be included as a control for its influence on network selection and to distinguish its effects from tolerance.


3. Model Specification: Translating Theory into Code
The RSiena model specification must precisely implement the proposed micro-theory. This involves two main components: network selection and attitude influence.

3.1. Selection Effects (How Tolerance Shapes Cooperation)


Primary Hypothesis: The core selection mechanism to be coded is the effect of an actor's tolerance on their likelihood of forming cooperation ties with ethnic out-group members. This should be implemented as an interaction effect (e.g., 


egoX tolerance interacting with same ethnicity on the cooperation network).


Constraint: The theory explicitly states that tolerance is not expected to drive the formation of friendships. Therefore, do not include a direct effect of tolerance on friendship tie formation.


Control Effects: The model must include standard controls for both the friendship and cooperation networks:

Endogenous network effects: Outdegree, reciprocity, and transitivity (e.g., GWESP) are mandatory.

Homophily effects: Control for homophily based on ethnicity and gender.

Prejudice effect: The effect of ego's and alter's prejudice on tie formation must be included as a key control.

3.2. Influence Effects (How Friendship Shapes Tolerance)

This is the most technically demanding part of the implementation.


Core Mechanism: The model must implement an attraction-repulsion mechanism for social influence on the tolerance attitude.



Constraint 1: Influence from Friends Only. The theory posits that influence stems from nominated friends, not the entire classroom. The standard attraction-repulsion effects available in 


RSiena (per Tang et al., 2025) are based on the average attitude of all actors in the group.

Action Item: You must write a new C++ effect for RSiena that calculates the attraction-repulsion influence term based only on the attitudes of an actor's outgoing friendship ties.


Constraint 2: Complex Contagion. The experimental design requires testing the difference between simple and complex contagion.


Action Item: You must program an additional custom C++ effect that models influence as a complex contagion. This effect should modify the friend-based attraction-repulsion mechanism so that influence is only exerted if an actor receives social reinforcement from multiple friends simultaneously. For example, the effect could be null unless at least 


k friends have a similar attitude.

4. The ABM Simulation Experiment: Design and Logic
The main purpose of the code is to run the computational experiment.


Calibration Phase: The code must first estimate the SAOMs on the empirical data for the 105 classes to obtain empirically calibrated parameters for the simulation phase. The code should be designed to handle the uncertainty of which parameters to use (Doubt 2); initially, create a workflow that can run simulations based on the parameters from one specific, well-fitting class.


Simulation Phase: The code will execute a large loop iterating through the different intervention designs. For each design:


Initialization: Start with the Wave 1 data (networks and attributes) of the chosen class.




Intervention: Programmatically modify the tolerance scores of a subset of majority-group actors according to the experimental conditions.



Targeting Logic (who): The code must be able to select target actors based on:

Popularity (high in-degree in the friendship network).

Other centrality measures (closeness, betweenness, eigenvector).


Peripheral position (e.g., low degree).

Varying Parameters: The main loop should iterate through:

Targeting strategy (

who to target).

Intervention size (

how large a subset).

Intervention intensity (

tolerance change).

Contagion type (simple vs. complex, using the custom effects).

Delivery strategy (random vs. clustered).

Execution: Run the siena07 simulation function with the modified data and calibrated parameters.

Output: For each run, save the final state of the system, including the complete cooperation network and the vector of tolerance scores.

5. Final Outputs for Analysis
The script's final output should be a single, tidy data frame. Each row should represent one complete simulation run. Columns should include:

All input parameters for that run (e.g., targeting_strategy, intervention_size, contagion_type).

Key summary statistics from the final state (e.g., mean tolerance, variance in tolerance, density of interethnic cooperation ties, network centralization).

This structured output will enable the final analysis and visualization for the paper. Please ensure the code is well-commented, especially the custom C++ components, to facilitate review and future extensions.


Sources






