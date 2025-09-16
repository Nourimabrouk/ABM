AI Agent Task Protocol: RSiena SAOM for Tolerance Intervention
Objective: To implement and execute a Stochastic Actor-Oriented Model (SAOM) in RSiena that simulates the effect of a targeted tolerance intervention. The goal is to determine the required magnitude of an intervention "shock" to produce a meaningful increase in network-wide average tolerance and the density of interethnic cooperation ties.

Phase 1: Data Preparation & Network Definition
Confirm Network Boundary: This is the most critical first step. You must determine if the friendship nomination data is:

A) Class-Bounded: Students could only nominate friends from within their own class.

B) School-Bounded: Students could nominate friends from their entire school year/grade.

Construct sienaData Objects:

If Class-Bounded: Create a list of 105 sienaData objects, one for each classroom. The analysis will later require a multi-level meta-analysis of the results from these 105 simulations (sienaBayes or similar).

If School-Bounded: Create 3 sienaData objects, one for each school. In this case, "class" becomes a categorical node attribute.

Define Dependent Variables: For each sienaData object, ensure the following dependent variables are correctly specified:

friendship: The friendship network (dyadic).

cooperation: The interethnic cooperation network (dyadic).

tolerance: The tolerance score (continuous behavior, scale 1-5).

Phase 2: Model Specification (getEffects)
Create a sienaEffects object. The core theoretical mechanisms must be included:

Friendship Network Dynamics:

outdegree (density)

reciprocity

transitiveTriplets (or gwespFF)

sameX on ethnicity (ingroup homophily)

simX on tolerance (selection based on tolerance similarity)

Cooperation Network Dynamics:

outdegree (density)

reciprocity

X effect for friendship (Do friends have a higher baseline probability of cooperating?)

Tolerance Attitude Dynamics:

linear and quadratic shape effects (tendency of the attitude).

Conditional Social Influence (Attraction-Repulsion): This is the key implementation. Use includeInteraction to make the Tang et al. attraction-repulsion effect conditional on a friendship tie.

myeff <- setEffect(myeff, attractionRepulsion, name = "tolerance", interaction1 = "friendship", include = FALSE)

myeff <- setEffect(myeff, crprod, name = "tolerance", interaction1 = "friendship", include = FALSE)

myeff <- includeInteraction(myeff, attractionRepulsion, crprod, name = "tolerance", interaction1 = c("friendship", "friendship"))

This ensures influence only happens between friends, as per your theory, without modifying C++ source code.

Phase 3: Experimental Simulation Protocol (siena07)
The core of the experiment is to run simulations under different intervention scenarios.

Define Intervention Magnitudes: Create a vector of shock values to test.

shock_magnitudes <- c(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, ...)

Create an Execution Loop: Loop through each sienaData object (whether 105 classes or 3 schools) and then loop through each value in shock_magnitudes.

Inside the Loop (For each combination of network and shock value):

a. Identify Targets: Calculate the in-degree for all actors in the friendship network at Wave 1. Select the top k actors (e.g., top 10%) as the intervention targets.

b. Apply Shock: Create a temporary copy of the Wave 1 tolerance data. For the target actors, update their score: tolerance 
new
​
 =min(5,tolerance 
old
​
 +shock_magnitude).

c. Create Experimental Data Object: Create a new sienaData object where the only change is this modified Wave 1 tolerance vector.

d. Run Simulation: Execute siena07 on this new experimental data object with the pre-defined effects object.

e. Store Results: Save the key outputs: the estimated parameters from the model and, crucially, the full simulated network and behavior data from the final state (sims object in the sienaFit return).

Phase 4: Analysis of Results
After the simulation loops are complete, process the stored results.

Calculate Success Metrics: For each simulation run, extract the final state data and compute:

Metric 1: The network-wide average tolerance score.

Metric 2: The density of ties in the cooperation network between actors of different ethnicities.

Visualize the Dose-Response Curve: Create a plot with:

X-axis: Intervention Magnitude (shock_magnitudes).

Y-axis: The resulting success metric (e.g., Average Tolerance).

This plot will reveal the minimum shock magnitude required to achieve a meaningful outcome, directly answering your primary research question.

This protocol provides a complete workflow from data setup to final analysis. We can begin implementation starting with Phase 1.







