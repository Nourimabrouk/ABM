On the Design of a Social Norm Intervention to Promote Interethnic Cooperation through Tolerance: An Agent-Based Modeling Approach
Researcher: Jan-Willem Simons


Collaborators: Jochem Tolsma, Eva Jaspers 


Affiliation: Department of Sociology, Utrecht University 

1. Background and Research Motivation
There is a significant and growing academic interest in designing interventions to improve intergroup attitudes and behaviors, particularly between members of different ethnic groups. A comprehensive review of the literature by Paluck, Porat, Clark & Green (2021) reveals a predominant focus within this field: existing interventions almost exclusively target the reduction of individual-level prejudice as the primary mechanism for achieving these improved outcomes. While prejudice is undoubtedly a critical factor, my research is motivated by two key criticisms of this narrow approach.


1.1. Beyond Prejudice: The Role of Principled Disapproval and Tolerance

The first criticism is that a sole focus on prejudice is insufficient because it overlooks the diverse motivations people have for structuring their social relationships based on ethnic group membership. Group-based prejudice, defined as the differential treatment of others based on their out-group status, is a major motivation. However, another distinct motivation is the principled disapproval of an out-group's beliefs, values, or practices. This disapproval can negatively impact interethnic relations independently of prejudice.



For instance, an individual might oppose the practice of Muslim women wearing headscarves not because of a prejudicial animus towards Muslims, but because they hold a strong personal or ideological commitment to gender equality norms. This illustrates a crucial point: individuals who are not prejudiced can still be intolerant and act on that intolerance. While the motivation may be more morally justifiable than one rooted in prejudice, the outcome can still be a deterioration of interethnic social cohesion.



In such cases, interventions should not target the disapproval itself, which may stem from deeply held values. Instead, they should promote value-based reasons to accept these practices despite the disapproval. This calls for interventions designed to foster 


tolerance, which can be defined as a behavioral intention to not interfere with a disapproved-of belief, value, or practice. To more holistically promote social cohesion, interventions must therefore not only address prejudice but also account for these reasonable sources of disapproval and manage their consequences by promoting tolerance.



1.2. Beyond the Individual: The Influence of Social Structure

The second criticism is that most interventions concentrate exclusively on enacting individual-level change, for example, through cognitive and emotional training designed to decrease personal prejudice. This approach largely ignores the social structures in which individuals are embedded. These structures—such as friendship networks and social norms—profoundly shape whether changed attitudes persist over time and whether they translate into improved behavior.



The limited size and duration of intervention effects observed in real-world settings, as compared to controlled laboratory environments, can likely be attributed to this oversight. Change does not occur in a social vacuum; individuals are always situated within social structures that influence how change unfolds. This points to the need for a more relational and longitudinal inquiry into how interventions can lead to sustained improvements in interethnic dynamics.



2. Research Objective
Given these two fundamental criticisms, the objective of my research is to produce initial insight into the following question: 

How can individual-level changes in tolerance, resulting from a hypothetical intervention, (1) spread and persist within a social network and (2) increase interethnic cooperation over time, given (3) different intervention designs? 

Put differently, my goal is to identify the design principles under which a tolerance-based intervention might be effective in promoting real-world interethnic cohesion, which I define as increased and sustained interethnic tolerance and cooperation.


To narrow the analytical focus and ground the model in a realistic context, I will specify the research context to align with the only known real-world intervention that aimed to promote ethnic out-group tolerance within a social network: a study of students in German high schools conducted by Shani et al. (2023) . Specifically, the model will concentrate on increasing tolerance among ethnic majority members (e.g., Native Germans) toward the practices, values, and beliefs of ethnic minority members (e.g., Turkish Germans).



3. A Proposed Micro-Theory of Tolerance and Cooperation
To simulate this process, I must first formulate a micro-theory that specifies the mechanisms through which increased tolerance can persist in a network and lead to greater cooperation. This theory involves three core concepts: tolerance as an individual attitude, and friendship and cooperation as social network relationships.

3.1. The Intervention and the Spread of Tolerance (Social Influence)

The model begins with a hypothetical intervention that increases an individual's tolerance by fostering 

equality-based respect as a counterweight to their disapproval. Equality-based respect is defined as "recognizing and treating outgroup individuals as equals, with the same rights and dignity as oneself". This choice is based on empirical evidence suggesting respect is a key psychological component of tolerance. A key assumption here is that a degree of disapproval is already present; otherwise, there would be nothing to tolerate.




Next, I theorize that this newly increased tolerance spreads through 

social influence, which is the process by which individuals adjust their attitudes through interaction with others. I will focus specifically on the influence of friends, as this mechanism has strong theoretical and empirical grounding for interethnic attitudes. The specific mechanism of influence will be an 



attraction-repulsion model , based on Social Judgment Theory. This model posits that:


Individuals will align their attitudes with their friends if the attitude difference is small (within a "latitude of acceptance").

No shift will occur if the difference is moderate.

Individuals will polarize and move their attitudes further away if the difference is sufficiently large.

While empirical support for this exact mechanism is mixed, it is an attractive theoretical option because it allows for the possibility of repulsion (i.e., friends causing each other to become less tolerant), not just assimilative influence.

3.2. The Behavioral Consequences of Tolerance (Social Selection)

After an individual becomes more tolerant, how does this affect their behavior in forming or maintaining relationships? This process is known as 

social selection. Theoretically, increased tolerance should primarily result in a decrease in negative out-group behaviors (e.g., bullying), as individuals now refrain from acting on their disapproval.


While reducing negative ties is a positive outcome, my research is primarily interested in whether tolerance can promote positive ties. The theoretical backing for this is admittedly sparse. It is unlikely that increased tolerance directly leads to interethnic 


friendship, as respect is not the same as appreciation, and tolerance is often a passive state without a clear approach motivation.

Instead, I postulate that increased tolerance can lead to increased interethnic 

cooperation. The theoretical argument, which has some empirical support, is that tolerance widens an individual's 

"radius of trust" to include out-group members. Assuming there is practical utility in cooperating with out-group members in a classroom setting (e.g., for schoolwork), an individual who becomes more tolerant will be more likely to trust and thus cooperate with an ethnic out-group member.



3.3. Summary of Micro-Theory and Controls

The micro-theory can be summarized by two core processes:


Influence: Friendship with ethnic in-group members influences an individual's level of tolerance via an attraction-repulsion mechanism.


Selection: An individual's level of tolerance influences their probability of cooperating with ethnic out-group members.

To isolate these effects, the model will also control for the confounding role of 

prejudice. Additionally, it will account for several standard endogenous network effects (e.g., reciprocity, transitivity) and an exogenous covariate (gender homophily).


4. Methodology: Agent-Based Modeling
The method I will use to test these theoretical propositions is an 

Agent-Based Model (ABM), specifically a Stochastic Actor-Oriented Model (SAOM). A SAOM is a powerful simulation tool that allows me to specify the micro-theoretical processes described above and examine their consequences using empirically calibrated parameters.


It serves as a "forecasting tool" to explore how micro-level mechanisms produce different macro-level outcomes (i.e., levels of interethnic cohesion) under various conditions. The crucial feature of this method is its ability to model the micro-macro link and study the emergent, network-level effects that arise from an initial change in individual tolerance.


The impetus for this modeling approach comes from the real-world intervention by Shani et al. (2023), which was found to have limited effectiveness. While there could be many reasons for this, I will proceed from the assumption that the intervention's design was a key factor and that an alternative design might have been successful. Rather than running thousands of impractical real-world experiments, I will use an ABM to simulate these different designs efficiently.





4.1. Experimental Design of the ABM

My ABM will have an experimental design where I systematically vary several key aspects of the intervention strategy:


Tolerance Change: What is the magnitude of tolerance change required to produce meaningful outcomes? 

Targeting Strategy:


Coverage: How large should the subset of targeted individuals be? 


Position: Should the intervention target popular students ("social referents") or those on the periphery ("norm entrepreneurs")? Should targets be selected based on network centrality measures like closeness, betweenness, or eigenvector centrality? 



Contagion Type: How do outcomes change if social influence is modeled as a complex contagion (requiring multiple exposures from friends to adopt the new attitude) versus a simple one? 



Delivery Strategy: Should the intervention be delivered to individuals in a clustered or random fashion within the network? 

4.2. Data and Simulation Plan

To ground the model, I will use data from a recently completed follow-up study to Shani et al. (2023) . This dataset contains 5,825 observations for 2,585 respondents in 105 classes across 3 schools, with 3 waves of data collection.


The simulation plan is as follows:

Fit SAOMs to the empirical data to obtain calibrated parameters for the micro-theory.

Manipulate the initial conditions of the model according to each cell in the experimental intervention matrix described above.

Simulate the evolution of the social network and attitudes over time for each scenario, examining the resulting outcomes in network-level tolerance and interethnic cooperation.

5. Current Doubts and Challenges
As I proceed with this research, I am grappling with several technical, methodological, and theoretical challenges that I would value your input on.


Doubt 1 (Technical): The existing attraction-repulsion model in the RSiena software package refers to influence from all students in a classroom, not specifically from one's friends. Implementing my proposed mechanism will require programming new effects in the C++ architecture of RSiena. I will also need to program a novel version of this effect to account for complex contagion dynamics.




Doubt 2 (Methodological): The data is heavily nested, with students in 105 classes within 3 schools. If the micro-theory finds empirical support at the class level, I am unsure of the best way to proceed with the simulations. Should I simulate from each class individually as a set of case studies? Should I pick one representative class? Or would it be better to fit a meta-analysis across the different SAOM estimates to generate a single, combined set of parameters to simulate from? 





Doubt 3 (Contingency): There is a risk that the proposed micro-theory will not be well-supported by the data. If I find support in only a few classes after an extensive model search, there would be a concern of ad hoc fitting. If I find no support at all, I would need to decide whether to formulate and test an alternative micro-theory, perhaps one focusing on the reduction of negative ties instead of the promotion of positive ones.




Doubt 4 (Theoretical): I find the theoretical argument that tolerance increases cooperation by expanding the "radius of trust" to be insufficiently specified. What exactly is the cognitive or emotional process that connects becoming more tolerant of a group's practices to becoming more trusting of individuals from that group in a cooperative setting?  I would be grateful for any relevant literature or theoretical insights you might have on this specific mechanism.


