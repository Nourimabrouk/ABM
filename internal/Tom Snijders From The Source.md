# Tom Snijders' PhD guidance ecosystem in network analysis

Tom A.B. Snijders, Emeritus Professor at Oxford and Groningen, has developed the most comprehensive training infrastructure for PhD students in longitudinal network analysis, centered on his groundbreaking Stochastic Actor-Oriented Models (SAOMs) and the RSiena software package. This research reveals an extensive ecosystem of resources, methodological innovations, and pedagogical materials that constitute the gold standard for doctoral training in network dynamics.

## The architecture of advanced PhD training

Snijders' approach to PhD supervision reflects a unique synthesis of mathematical rigor, software implementation, and practical application. His **Oxford statistics page** (https://www.stats.ox.ac.uk/~snijders/) serves as the central hub for accessing resources, including the comprehensive **200+ page RSiena manual** that undergoes continuous updates. The manual, co-authored with Ruth Ripley, Zsófia Boda, András Vörös, and Paulina Preciado, represents the definitive technical reference for SAOM implementation. His teaching philosophy emphasizes hands-on learning through real data, with students progressing from basic network concepts to advanced multilevel specifications through carefully structured exercises.

The crown jewel of his pedagogical resources is the **18-session online workshop series** available at InStats.org, where each 45-minute session builds systematically from foundational concepts to cutting-edge techniques. This self-paced format allows PhD students worldwide to access Oxford-quality instruction on demand. The workshop progression follows a deliberate arc: starting with data structures and basic SAOM theory, advancing through network-behavior co-evolution, and culminating in multilevel Bayesian approaches using the sienaBayes function.

## State-of-the-art coding practices and technical mastery

Snijders' technical guidance establishes strict standards for computational excellence in network analysis. His **convergence criteria** demand overall maximum convergence ratios below 0.25, with individual t-ratios preferably under 0.1. This precision extends to his recommended workflow, which begins with minimal models (density and reciprocity) before systematically adding structural effects like transitivity (transTrip) and three-cycles (cycle3). The approach prevents common specification errors through forward selection procedures that avoid collinearity while maintaining theoretical coherence.

The **GitHub repository** (https://github.com/stocnet/rsiena) showcases active development with both stable and experimental branches, demonstrating best practices in reproducible research. Advanced users can access the RSienaTest package through R-Forge for experimental features, while the main CRAN version ensures stability for production research. His coding standards emphasize proper data structures—networks as three-dimensional arrays, covariates specified as constant or changing, and careful handling of missing data through multiple imputation rather than listwise deletion.

Recent technical innovations include the **Stochastic Actor-Oriented Model with Random Effects** (2024), addressing individual heterogeneity through extended method-of-moments estimation. This breakthrough, published with Giacomo Ceoldo and Ernst Wit, represents a fundamental advance in accounting for unobserved actor-level variation in network dynamics.

## Workshop materials from Sunbelt and beyond

The **Sunbelt 2023 Portland workshop** introduced advanced topics including multivariate networks, two-mode/one-mode co-evolution, valued networks with weak/strong ties, and semi-standardized parameter interpretation. Materials from this workshop, available at https://www.stats.ox.ac.uk/~snijders/siena/Workshop_Sunbelt_2023.htm, include detailed slides, R scripts, and literature references focusing on multilevel longitudinal network estimation using sienaBayes.

The **University of Exeter online workshop** (August 2020) provides the most comprehensive publicly available resource, featuring 12 slide sets (A through L) covering everything from basic network dynamics to multilevel Bayesian analysis. Each module includes presentation slides, printable handouts, and corresponding R scripts. Module D, "Model Specification Recommendations for Siena," distills decades of experience into practical guidelines for avoiding common pitfalls. Module K introduces sienaBayes for multilevel network analysis, representing the frontier of methodological development.

The annual **Groningen Winter School on Longitudinal Social Network Analysis** enters its 15th edition in January 2025, offering three days of intensive training followed by the Advanced Siena Users Meeting (AdSUM). This format allows progression from foundational concepts to cutting-edge techniques, with participants presenting their own research for feedback from Snijders and international colleagues.

## Common mistakes and methodological wisdom

Snijders identifies critical errors that derail SAOM research, most notably the **circular specification problem** documented with Per Block, James Hollway, and Christoph Stadtfeld. This occurs when researchers use nodal covariates calculated from observed degrees instead of proper structural effects, "turning endogeneity into circularity" and rendering predictions tautological. His stark advice: never use future information to predict the future.

Time heterogeneity represents another frequent oversight. Rather than assuming temporal stability, Snijders advocates statistical testing using the sienaTimeTest function, with proper diagnostic procedures to identify period-specific effects. His workshop materials emphasize the distinction between selection and influence mechanisms, advocating for selection tables and influence tables that clarify causal pathways in network-behavior co-evolution.

Model specification errors plague novice researchers who add effects indiscriminately. Snijders recommends systematic forward selection, starting with theoretically motivated effects and adding complexity only when justified by substantive theory and convergence diagnostics. The goodness-of-fit testing framework, implemented through sienaGOF, provides essential validation that many researchers overlook.

## PhD supervision philosophy and success stories

Snijders' supervision philosophy combines methodological rigor with practical implementation, producing an academic lineage that reads like a who's who of network analysis. Former students including Christian Steglich (now RSiena's maintainer), Miranda Lubbers, Per Block, and Nynke Niezink occupy prominent positions internationally. This success reflects his emphasis on integrating theoretical understanding with software development—students don't just learn methods, they contribute to their evolution.

His approach to mentoring emphasizes collaborative research, with many publications featuring student co-authors. The ICS (Interuniversity Center for Social Science Theory and Methodology) provides institutional support for this model, connecting students across Groningen, Utrecht, and Nijmegen in a structured doctoral program. Recent students work on diverse applications from criminal network disruption to corporate environmental governance, reflecting the breadth of SAOM applications.

## Advanced techniques and unpublished innovations

The **sienaBayes function** represents Snijders' most sophisticated methodological contribution, enabling Bayesian multilevel network analysis with random coefficients. This approach, detailed in workshop materials from Sunbelt 2024, allows researchers to model heterogeneity across multiple networks simultaneously. The method addresses a fundamental limitation of traditional SAOMs by accounting for group-level variation in network dynamics.

Recent workshop materials reveal unpublished techniques for handling **non-convergence** through algorithmic adjustments. Setting diagonalize=0.2 reduces instability in poorly conditioned models, while the doubleAveraging parameter provides alternative convergence pathways. For particularly challenging models, finite difference approximations (findiff=TRUE) can stabilize estimation when analytical derivatives fail.

The **selection and influence decomposition** using Moran statistics, presented in advanced workshops, offers a principled approach to separating social selection from social influence in co-evolutionary models. This technique, combined with entropy-based measures of explained variation, provides interpretable effect sizes that address longstanding criticisms of SAOM parameters.

## Future trajectories and cutting-edge developments

Snijders envisions network analysis evolving toward greater integration with other methodological paradigms. His recent work on **relational event models** (REMs) with Viviana Amati and Alessandro Lomi establishes goodness-of-fit frameworks for continuous-time network processes. The forthcoming integration of REMs with SAOMs promises to bridge discrete and continuous temporal perspectives.

The **StOCNET ecosystem** now encompasses six R packages—RSiena, manynet, migraph, goldfish, MoNAn, and ERPM—representing a comprehensive toolkit for network dynamics. This modular architecture enables researchers to combine methods flexibly, moving beyond single-model approaches toward integrated analytical frameworks.

Two-mode network co-evolution, featured in his 2025 Sunbelt presentation, extends SAOMs to bipartite structures with implications for organizational research, ecological networks, and recommender systems. The ability to model simultaneous evolution of affiliation and interaction networks opens new frontiers in understanding multilevel social processes.

## Computational resources and implementation standards

Snijders provides explicit guidance on computational optimization, recommending parallel processing (useCluster=TRUE, nbrNodes=4) for production runs while maintaining reproducibility through seed management. For initial exploration, reduced iterations (nsub=2, n3=100) enable rapid prototyping, while final models require full specifications (nsub=4, n3=1000) for publication-quality results.

Memory limitations constrain network size to several hundred actors, though recent optimizations extend this boundary. His documentation emphasizes the trade-off between model complexity and computational feasibility, advocating parsimonious specifications that balance theoretical richness with practical constraints. The workshop materials include benchmarking scripts that help researchers estimate computation time for different model specifications.

## Access to resources and continuing education

All major resources remain freely accessible through Snijders' Oxford page, the RSiena website, and GitHub repositories. The commitment to open science extends to workshop materials, with slides, scripts, and datasets available for self-study. The recorded InStats seminar series provides structured learning paths, while annual workshops offer opportunities for direct interaction.

The Advanced Siena Users Meeting (AdSUM) represents the pinnacle of continuing education, where experienced researchers present challenging applications for collective problem-solving. This format exemplifies Snijders' collaborative approach to methodological development, where user experiences inform software evolution and theoretical refinements.

His recent receipt of the **Paul F. Lazarsfeld Award** (2022) from the American Sociological Association recognizes these contributions to sociological methodology. Yet Snijders continues pushing boundaries, with 2024 publications on random effects models and goodness-of-fit frameworks demonstrating ongoing innovation at 75 years of age.

## Conclusion

Tom Snijders has constructed an unparalleled infrastructure for doctoral training in network analysis, combining rigorous mathematical foundations with practical software implementation and extensive pedagogical resources. His guidance encompasses technical mastery, methodological sophistication, and philosophical wisdom accumulated over four decades of pioneering research. The resources available—from comprehensive manuals to cutting-edge workshop materials—provide PhD students with everything needed to conduct state-of-the-art longitudinal network analysis. His continuing innovations in random effects models, multilevel approaches, and software development ensure that students trained in his methods remain at the forefront of network science.