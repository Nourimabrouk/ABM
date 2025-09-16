# Agent Interaction Template

Use this template for structured communication between agents in the ABM tolerance intervention research project.

## Communication Structure

### Standard Agent Interaction Format

```
TO: [Target Agent(s)]
FROM: [Sending Agent]
TYPE: [Collaboration|Consultation|Review|Decision|Update]
PRIORITY: [Critical|High|Medium|Low]
CONTEXT: [Brief description of situation/task]

OBJECTIVE:
[Clear statement of what is needed from the interaction]

BACKGROUND:
[Relevant context and prior work]

SPECIFIC REQUESTS:
1. [Specific task or question]
2. [Additional requirements]
3. [Timeline expectations]

DELIVERABLES:
- [Expected outputs]
- [Format requirements]
- [Quality standards]

DEPENDENCIES:
[What this depends on or blocks]

TIMELINE:
[Deadlines and milestones]
```

## Interaction Types

### 1. Technical Collaboration
```
EXAMPLE: SAOM Specialist → Simulation Engineer

TO: Simulation_Engineer
FROM: SAOM_Specialist  
TYPE: Collaboration
PRIORITY: High
CONTEXT: Custom C++ effect performance optimization

OBJECTIVE:
Optimize friend-based attraction-repulsion effect for large-scale simulations

BACKGROUND:
Initial C++ implementation shows O(n²) complexity per actor update. With 2,585 students across multiple simulation runs, this creates computational bottleneck.

SPECIFIC REQUESTS:
1. Review current algorithm for performance optimization opportunities
2. Suggest efficient data structures for friend network queries
3. Implement parallel processing for independent actor updates
4. Benchmark against target performance (<30 seconds per simulation)

DELIVERABLES:
- Optimized C++ code with complexity analysis
- Performance benchmarks comparing old vs. new implementation
- Documentation of optimization strategies
- Integration testing results

DEPENDENCIES:
Blocks parameter sweep execution until performance meets requirements

TIMELINE:
Initial review: 2 days
Optimization implementation: 5 days
Testing and validation: 3 days
```

### 2. Academic Consultation
```
EXAMPLE: Statistical Analyst → Tolerance Theory Expert

TO: Tolerance_Theory_Expert
FROM: Statistical_Analyst
TYPE: Consultation
PRIORITY: Medium
CONTEXT: Effect size interpretation for tolerance-cooperation pathway

OBJECTIVE:
Establish substantively meaningful effect size benchmarks for tolerance interventions

BACKGROUND:
Statistical analysis shows intervention effects ranging from d=0.2 to d=0.8 across different parameter combinations. Need theoretical grounding for interpreting practical significance.

SPECIFIC REQUESTS:
1. Review literature for typical tolerance intervention effect sizes
2. Advise on minimum meaningful change in tolerance attitudes
3. Assess realistic expectations for cooperation behavior change
4. Provide boundary conditions for intervention effectiveness

DELIVERABLES:
- Literature review summary with effect size benchmarks
- Theoretical justification for meaningful change thresholds
- Interpretation guidelines for statistical results
- Recommendations for practical significance criteria

DEPENDENCIES:
Required for final results interpretation and discussion

TIMELINE:
Literature review: 4 days
Interpretation framework: 2 days
```

### 3. Quality Review
```
EXAMPLE: Research Methodologist → All Agents

TO: All_Agents
FROM: Research_Methodologist
TYPE: Review
PRIORITY: Critical
CONTEXT: Mid-project quality assurance checkpoint

OBJECTIVE:
Comprehensive review of all work products to ensure PhD-level academic standards

BACKGROUND:
Approaching 50% project completion. Need systematic review to identify any methodological issues before proceeding to final analysis phase.

SPECIFIC REQUESTS:
1. Submit all current deliverables for methodological review
2. Provide rationale for key design decisions
3. Identify any deviations from original research plan
4. Assess reproducibility of current work

DELIVERABLES:
- Complete work portfolio for review
- Methodology justification documents
- Reproducibility verification checklist
- Quality improvement recommendations

DEPENDENCIES:
Critical gate for proceeding to final analysis phase

TIMELINE:
Submission deadline: 3 days
Review completion: 5 days
Revision implementation: 7 days
```

### 4. Decision Making
```
EXAMPLE: Multi-Agent Decision Process

TO: SAOM_Specialist, Statistical_Analyst, Tolerance_Theory_Expert
FROM: Research_Methodologist
TYPE: Decision
PRIORITY: High
CONTEXT: Meta-analysis approach for 105-class dataset

OBJECTIVE:
Determine optimal strategy for combining results across multiple school classes

BACKGROUND:
SAOM estimation successful in 78 of 105 classes. Need decision on whether to: (1) meta-analyze across successful classes, (2) focus on subset of best-fitting classes, or (3) implement multilevel modeling approach.

SPECIFIC REQUESTS:
1. SAOM_Specialist: Assess technical feasibility of each approach
2. Statistical_Analyst: Evaluate statistical implications and power
3. Tolerance_Theory_Expert: Consider theoretical appropriateness
4. All: Reach consensus recommendation with rationale

DELIVERABLES:
- Technical feasibility assessment
- Statistical power analysis for each option
- Theoretical coherence evaluation
- Consensus recommendation with implementation plan

DEPENDENCIES:
Blocks final analysis strategy and timeline

TIMELINE:
Individual assessments: 3 days
Consensus meeting: 1 day
Implementation planning: 2 days
```

## Quality Standards for Interactions

### Communication Excellence
- **Clarity**: Precise and unambiguous requests
- **Completeness**: All necessary context provided
- **Specificity**: Concrete deliverables and timelines
- **Professionalism**: Academic tone and standards

### Academic Rigor
- **Evidence-Based**: All claims supported by data or literature
- **Methodologically Sound**: Requests align with research standards
- **Theoretically Grounded**: Academic foundation for all work
- **Ethically Appropriate**: Responsible research practices

### Collaborative Effectiveness
- **Respectful**: Professional and constructive tone
- **Efficient**: Clear priorities and realistic timelines
- **Integrated**: Recognition of interdependencies
- **Solution-Oriented**: Focus on advancing research objectives

## Response Template

```
TO: [Original Sender]
FROM: [Responding Agent]
RE: [Original Subject]
STATUS: [Complete|In Progress|Needs Clarification|Blocked]

SUMMARY:
[Brief overview of response]

RESPONSES TO SPECIFIC REQUESTS:
1. [Address each numbered request]
2. [Provide concrete answers/deliverables]
3. [Include rationale for recommendations]

DELIVERABLES PROVIDED:
- [List actual outputs]
- [Include location/format information]

ADDITIONAL INSIGHTS:
[Any relevant observations or recommendations]

FOLLOW-UP NEEDED:
[What additional work is required]

TIMELINE UPDATE:
[Any changes to expected completion]
```

## Emergency Communication Protocol

### Critical Issue Escalation
```
TO: All_Agents
FROM: [Reporting Agent]
TYPE: Emergency
PRIORITY: Critical
CONTEXT: [Nature of critical issue]

ISSUE DESCRIPTION:
[Clear description of problem]

IMMEDIATE IMPACT:
[Effect on research progress/quality]

PROPOSED RESOLUTION:
[Suggested approach to address issue]

REQUIRED ASSISTANCE:
[Specific help needed from other agents]

TIMELINE:
[Urgency for resolution]
```

---

*This template ensures consistent, professional, and efficient communication across all agents while maintaining the high academic standards required for PhD-level research.*