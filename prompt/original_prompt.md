# The original prompt (phase 1), verbatim

Sent 2026-09-04 23:51 UTC to Claude Fable 5.1 running as an autonomous agent in Claude Code (Anthropic desktop app) on an
Apple M4 Max, with five reference images attached (`../carbon_discovery/reference_images/`). Nothing below was edited.

---

Convert the supplied reference images into a complete, browser-based generative-design and scientific-discovery platform for atomically resolved carbon nanomaterials, focused on graphene and related two-dimensional carbon architectures.

The supplied images may represent different scales, geometries, abstractions, networks, hierarchical structures, organic morphologies, or design concepts. They are not necessarily registered views of the same object and should not be treated as literal projections of one structure, but use it to extract design languages of hierarchical materials.

Infer a scientifically meaningful design language from the images and translate it into explicit atomic carbon structures.

Do not simply trace image edges or place atoms along visible features.

Extract transferable geometric and topological concepts from the images and determine how they can be represented as physically meaningful carbon architectures.

The mechanical response must NOT come from:
- springs;
- beams;
- continuum approximations;
- generic rigid-body physics;
- manually deleted visual bonds;
- neural-network surrogate mechanics;
- visually simulated fracture.

The authoritative mechanics must come from an actual many-body classical carbon interatomic potential.

Use screened second-generation REBO as the primary production force field.

Use ASE as the canonical atomic structure representation and interoperability layer.

Where computationally feasible, implement GPU acceleration using PyTorch so that the same force engine can execute on:
- Apple Silicon through MPS;
- NVIDIA GPUs through CUDA;
- CPU as a numerical-reference and fallback mode.

Scientific correctness takes precedence over acceleration.

The ultimate objective is not merely to generate attractive carbon structures or optimize a scalar objective.

The objective is to autonomously discover mechanistically interpretable relationships between carbon architecture and fracture.

──────────────────────────────────────────────────────────────────────────────
CORE SCIENTIFIC QUESTION
──────────────────────────────────────────────────────────────────────────────

Investigate:

How does atomic-scale carbon architecture control stiffness, strength, failure strain, energy absorption, flaw sensitivity, crack initiation, crack propagation, damage localization, and the transition between abrupt and progressive fracture?

Treat structural complexity as something to investigate rather than something assumed to be beneficial.

Do not assume that:
- hierarchy improves mechanical performance;
- disorder produces toughness;
- redundancy improves fracture resistance;
- porosity is beneficial;
- crack deflection necessarily produces toughening;
- visually complex structures outperform simple ones;
- generated structures remain stable after atomic relaxation;
- the force field quantitatively reproduces experimental graphene fracture.

Null results, negative results, tradeoffs, unstable structures, counterexamples, and failed hypotheses are scientifically meaningful outcomes.

──────────────────────────────────────────────────────────────────────────────
REFERENCE-IMAGE INTERPRETATION
──────────────────────────────────────────────────────────────────────────────

Begin by examining the supplied images without assuming a predetermined structural interpretation.

Infer a small set of transferable geometric and topological design principles that can meaningfully be instantiated in atomically resolved carbon.

Possible concepts may include, but are not limited to:
- hierarchy;
- branching;
- nested structures;
- connectivity;
- loops;
- porosity;
- anisotropy;
- redundancy;
- ordered versus disordered organization;
- gradients;
- interfaces;
- tortuous pathways;
- geometric bottlenecks;
- scale transitions;
- local defects;
- load-path organization.

These are possibilities, not mandatory interpretations.

Do not force any particular concept onto an image unless the interpretation is defensible.

For each inferred principle explain:

1. what visual feature motivated the interpretation;
2. what abstract design principle was extracted;
3. how that principle can be represented atomistically;
4. what assumptions are required;
5. what alternative interpretations are plausible.

Expose this interpretation in an editable application panel.

Allow the user to modify the interpretation and regenerate structures.

──────────────────────────────────────────────────────────────────────────────
DEEP DESIGN-SPACE EXPLORATION
──────────────────────────────────────────────────────────────────────────────

The design space must be explored deeply rather than by generating one example of many unrelated structure classes.

Identify several major structural variables emerging from the image interpretation and systematically explore them.

HIERARCHY MUST BE ONE OF THE MAJOR DESIGN VARIABLES.

Treat hierarchy as an experimentally controlled variable, not as an assumed advantage.

Construct a hierarchy ladder that spans, where atomistically meaningful:

- no hierarchy / single-scale structure;
- weak or shallow hierarchy;
- two-level hierarchy;
- multiple hierarchical levels;
- strongly nested or multiscale organization.

Explore relevant hierarchical variables such as:

- hierarchy depth;
- characteristic size at each level;
- scale ratio between levels;
- nesting of pores or structural motifs;
- coarse-to-fine branching;
- inter-level connectivity;
- inter-level redundancy;
- orientation between levels;
- spatial localization of hierarchy;
- graded versus discrete hierarchy;
- hierarchical defect organization;
- hierarchical load-path organization.

The exact geometric realization of these variables should emerge from the interpretation of the supplied images rather than from a predefined library of hierarchical motifs.

Also investigate other important axes where scientifically appropriate, including:

- order → partial disorder → strong disorder;
- low → high redundancy;
- isotropic → strongly anisotropic structures;
- low → high porosity;
- different pore aspect ratios and orientations;
- uniform → graded geometry;
- distributed → clustered defects;
- simple → tortuous load paths;
- low → high connectivity;
- homogeneous → heterogeneous ligament dimensions.

Explore interactions between variables rather than only one-dimensional sweeps.

Examples include:

hierarchy × disorder

hierarchy × anisotropy

hierarchy × redundancy

hierarchy × porosity

hierarchy × defect organization

hierarchy × load-path alignment

Do not vary all parameters simultaneously.

Use controlled comparisons whenever possible.

When testing the effect of hierarchy or another architectural feature, create matched structures in which competing factors such as:
- carbon mass;
- porosity;
- footprint;
- defect density;
- or another relevant quantity

are controlled.

The objective is to identify mechanisms, thresholds, interactions, and regime transitions rather than merely correlations.

──────────────────────────────────────────────────────────────────────────────
ATOMIC REPRESENTATION
──────────────────────────────────────────────────────────────────────────────

Every simulated structure must be represented explicitly by:

- carbon atomic coordinates;
- simulation cell;
- atomic species;
- periodic/nonperiodic boundary conditions;
- metadata describing how the structure was generated.

Use ASE Atoms objects as the canonical representation.

Every generated material must be convertible to and from ASE without losing:

- atomic coordinates;
- cell;
- periodicity;
- design parameters;
- random seed;
- run identifier;
- relevant metadata.

Support export to:

- ASE trajectory;
- XYZ;
- Extended XYZ;
- LAMMPS data;
- CIF where meaningful;
- POSCAR where meaningful.

STL is not an atomistic representation.

Optional STL files may only represent macroscopically enlarged analogues of selected atomic structures and must be clearly labeled as such.

──────────────────────────────────────────────────────────────────────────────
CARBON ARCHITECTURE GENERATION
──────────────────────────────────────────────────────────────────────────────

Use pristine graphene as the fundamental reference structure.

From the image-derived design principles, generate several structurally distinct families of graphene-derived or related 2D carbon architectures.

Do NOT mechanically generate one example from a predetermined checklist.

Instead choose architecture classes because they instantiate the inferred design principles and allow meaningful scientific questions to be tested.

Possible transformations include:

- atom removal;
- pores;
- oriented pores;
- cuts;
- cracks;
- vacancies;
- correlated defects;
- ring/topology transformations;
- nanomeshes;
- graphene kirigami;
- grain structures;
- interconnected graphene ligaments;
- nested structures;
- hierarchical pore systems;
- ordered networks;
- disordered networks;
- graded architectures;
- anisotropic networks;
- topology-changing motifs.

These are examples rather than requirements.

Where known carbon structures such as graphyne-like materials are used, employ validated atomic templates.

Do not label arbitrary generated structures as known allotropes.

──────────────────────────────────────────────────────────────────────────────
ATOMIC PLAUSIBILITY AND STRUCTURAL VIABILITY
──────────────────────────────────────────────────────────────────────────────

A generated geometry is only a candidate.

Do not assume it represents a viable carbon material.

For every candidate:

1. construct the atomic structure;
2. check minimum atomic separation;
3. detect duplicate atoms;
4. examine connectivity;
5. record initial coordination statistics;
6. calculate initial energy;
7. relax using the primary force field;
8. examine reconstruction;
9. determine whether a mechanically continuous structure remains.

Record:

- energy before relaxation;
- energy after relaxation;
- maximum force;
- maximum atomic displacement;
- RMS displacement;
- coordination changes;
- new neighbor relationships;
- lost neighbor relationships;
- isolated atoms;
- fragments;
- spanning/load-bearing connectivity.

Classify the structure as:

- stable;
- reconstructed but viable;
- strongly reconstructed;
- unstable;
- fragmented;
- disconnected;
- failed during minimization;
- numerically unresolved.

Do not silently discard unsuccessful structures.

They define the boundaries of the design space and should remain in the experiment database.

──────────────────────────────────────────────────────────────────────────────
PRIMARY FORCE FIELD — SCREENED REBO2
──────────────────────────────────────────────────────────────────────────────

Use the published screened second-generation REBO carbon potential as the authoritative production force field.

This potential may be referred to as:

- screened REBO2;
- REBO2-S;
- REBO2+S;
- Rebo2Scr.

Use the established Atomistica Rebo2Scr implementation as the authoritative reference implementation unless a demonstrably equivalent established implementation is available.

Do not silently substitute:

- standard REBO2;
- AIREBO;
- AIREBO-M;
- Tersoff;
- modified Tersoff;
- harmonic bonding;
- Morse bonding;
- machine-learned force fields.

Do not modify published parameters, screening functions, or cutoff functions merely to obtain desired fracture behavior.

Record with every run:

- exact potential;
- implementation;
- parameterization/version;
- literature citation;
- source version or git commit;
- parameter checksum where meaningful;
- units.

The production discovery study should use one validated primary potential consistently.

Do not routinely spend the scientific campaign comparing many force fields.

Only investigate another potential if a validation problem, numerical anomaly, or scientific question discovered during the study provides a specific reason to do so.

──────────────────────────────────────────────────────────────────────────────
GPU-NATIVE IMPLEMENTATION
──────────────────────────────────────────────────────────────────────────────

A GPU-native PyTorch implementation of screened REBO2 is strongly preferred if it can be implemented exactly and validated.

A possible module name is:

TorchRebo2Scr

The implementation should support:

torch.device("mps")
torch.device("cuda")
torch.device("cpu")

with automatic backend detection.

The GPU implementation must reproduce the same screened REBO2 potential used by the authoritative implementation.

Do not simplify the functional form for convenience or speed.

Implement all relevant terms required by the published potential, including:

- repulsive interactions;
- attractive interactions;
- environment-dependent bond order;
- angular dependence;
- coordination/conjugation dependence where required;
- screening;
- cutoff behavior;
- periodic boundaries;
- energy;
- forces;
- virial/stress.

Automatic differentiation may be used for forces:

F = -∂E/∂R

provided correctness and computational efficiency are demonstrated.

If an exact GPU implementation cannot be completed and validated, use the authoritative screened REBO2 implementation for the scientific study.

Do NOT substitute a simpler potential merely to obtain MPS acceleration.

Scientific correctness has priority over GPU execution.

──────────────────────────────────────────────────────────────────────────────
FLOATING-POINT PRECISION
──────────────────────────────────────────────────────────────────────────────

Where PyTorch MPS is used, production calculations may require float32.

Maintain:

FAST DISCOVERY MODE
- MPS/CUDA
- float32

REFERENCE MODE
- CPU float64
or
- CUDA float64 where supported.

Quantify differences between fast and reference modes for:

- energy;
- forces;
- relaxed geometry;
- elastic modulus;
- strength;
- failure strain;
- fracture sequence.

Set numerical acceptance tolerances before the discovery campaign.

──────────────────────────────────────────────────────────────────────────────
NEIGHBOR LIST AND COMPUTATIONAL ENGINE
──────────────────────────────────────────────────────────────────────────────

Use an efficient short-range neighbor-list implementation.

Do not use an O(N²) distance matrix for large structures unless justified by system size.

Support:

- periodic dimensions;
- nonperiodic dimensions;
- 2D sheets;
- Verlet skin;
- cell-list or equivalent spatial decomposition;
- neighbor-list rebuilding.

Test the optimized neighbor list against brute-force enumeration for small systems.

Record:

- interaction cutoff;
- neighbor skin;
- rebuild criterion;
- average neighbor count;
- rebuild frequency.

Where batching is feasible, allow multiple independent structures to be evaluated concurrently.

──────────────────────────────────────────────────────────────────────────────
ASE WORKFLOW
──────────────────────────────────────────────────────────────────────────────

Use ASE for:

- graphene construction;
- supercells;
- atomic transformations;
- cells and boundary conditions;
- structure storage;
- trajectories;
- metadata;
- file I/O;
- interoperability with reference calculators.

ASE should not create a CPU bottleneck inside every GPU force evaluation.

Keep the inner accelerated simulation loop on the selected device where possible.

──────────────────────────────────────────────────────────────────────────────
FORCE-FIELD VALIDATION
──────────────────────────────────────────────────────────────────────────────

Validation is mandatory before scientific discovery begins.

Use Atomistica Rebo2Scr as the primary numerical oracle.

Test both ordinary and difficult configurations.

Include:

1. pristine graphene;
2. randomly displaced graphene;
3. strained graphene;
4. vacancies;
5. pore edges;
6. crack-tip configurations;
7. highly strained configurations near rupture.

Compare:

- total energy;
- energy per atom;
- every force component;
- force norms;
- virial/stress;
- relaxed structures.

──────────────────────────────────────────────────────────────────────────────
BOND-SEPARATION VALIDATION
──────────────────────────────────────────────────────────────────────────────

Before running the fracture campaign, explicitly validate the behavior of the potential during bond separation.

Construct representative carbon bonding environments and continuously increase a relevant separation coordinate from equilibrium through complete bond rupture.

Calculate and plot:

E(r)

and

F(r) = -dE/dr

Inspect for:

- force discontinuities;
- artificial force spikes;
- cutoff-induced re-stiffening;
- unexpected metastable states;
- numerical instability;
- disagreement with the authoritative implementation.

The accelerated implementation must reproduce the authoritative force and energy curves through the entire rupture process, not merely near equilibrium.

Do not proceed to the discovery campaign if bond-breaking behavior is incorrect.

──────────────────────────────────────────────────────────────────────────────
GENERAL VALIDATION SUITE
──────────────────────────────────────────────────────────────────────────────

At minimum include:

TEST 1 — energy against authoritative implementation

TEST 2 — forces against authoritative implementation

TEST 3 — finite-difference force check on a small structure

TEST 4 — MPS/CUDA versus CPU precision

TEST 5 — relaxed graphene lattice constant and bond length

TEST 6 — graphene cohesive energy

TEST 7 — small-strain elastic response

TEST 8 — armchair versus zigzag tensile response

TEST 9 — optimized versus brute-force neighbor list

TEST 10 — translation invariance

TEST 11 — total-force balance

TEST 12 — periodic-image invariance

TEST 13 — minimizer convergence

TEST 14 — strain-step convergence

TEST 15 — system-size convergence

TEST 16 — precision near fracture

TEST 17 — reproducibility

TEST 18 — complete stress-strain comparison with the reference backend for selected structures

TEST 19 — bond-separation validation

TEST 20 — export/data consistency

Display validation results in the application with:

- expected result;
- observed result;
- numerical error;
- tolerance;
- PASS/FAIL.

Do not begin autonomous scientific discovery if essential validation fails.

──────────────────────────────────────────────────────────────────────────────
SIMULATION MODES
──────────────────────────────────────────────────────────────────────────────

PRIMARY MODE — ATHERMAL QUASI-STATIC FRACTURE

Use AQS for the main large-scale design study.

At each loading step:

1. increment strain;
2. update cell or controlled boundaries;
3. relax atomic coordinates;
4. relax transverse cell dimensions where appropriate;
5. calculate stress;
6. record energy and structure;
7. detect persistent topology changes;
8. continue.

Use sufficiently small strain increments to resolve fracture events.

Where a fracture instability occurs, consider adaptive refinement of the strain increment.

Do not allow poor minimizer convergence to determine apparent fracture mechanisms without clearly identifying it.

OPTIONAL MODE — MOLECULAR DYNAMICS

Support finite-temperature MD when useful for:
- selected mechanism validation;
- temperature effects;
- rate effects;
- dynamic fracture.

Do not make finite-temperature MD mandatory for the full discovery campaign.

──────────────────────────────────────────────────────────────────────────────
BOUNDARY CONDITIONS
──────────────────────────────────────────────────────────────────────────────

Support relevant combinations of:

- fully periodic graphene;
- periodic loading direction;
- transverse relaxation;
- finite sheets;
- gripped finite sheets;
- nanoribbons;
- precracked structures.

Support:

- uniaxial tension;
- biaxial loading where scientifically useful;
- armchair loading;
- zigzag loading;
- arbitrary in-plane orientation.

Store the complete boundary condition with every simulation.

──────────────────────────────────────────────────────────────────────────────
2D STRESS AND UNITS
──────────────────────────────────────────────────────────────────────────────

Graphene is a two-dimensional material.

Use 2D membrane stress / force per width in N/m as the primary stress measure.

Where a conventional 3D-equivalent stress is reported, explicitly specify the assumed effective thickness.

Never silently convert N/m into GPa.

Report:

- 2D modulus;
- 2D strength;
- strain;
- optional clearly labeled 3D-equivalent quantities.

──────────────────────────────────────────────────────────────────────────────
STRUCTURAL DESCRIPTORS
──────────────────────────────────────────────────────────────────────────────

Calculate descriptors appropriate to each structure, potentially including:

- atom count;
- carbon mass;
- footprint;
- areal density;
- porosity;
- pore dimensions;
- pore anisotropy;
- ligament dimensions;
- defect density;
- defect clustering;
- coordination distribution;
- lattice orientation;
- topology;
- hierarchy depth;
- hierarchy scale ratio;
- hierarchy localization;
- inter-level connectivity;
- redundancy;
- anisotropy;
- disorder;
- path tortuosity;
- bottleneck size;
- load-path orientation.

Descriptors should help explain mechanisms, not merely provide features for a statistical model.

──────────────────────────────────────────────────────────────────────────────
MECHANICAL AND FRACTURE METRICS
──────────────────────────────────────────────────────────────────────────────

Calculate where meaningful:

- relaxed energy;
- energy per atom;
- initial 2D modulus;
- Poisson response;
- ultimate strength;
- strain at peak stress;
- first irreversible damage strain;
- failure strain;
- work to failure;
- work to failure per area;
- specific work to failure;
- residual post-damage load;
- number of major damage events;
- crack-initiation position;
- crack direction;
- crack path;
- crack tortuosity;
- crack branching;
- fraction of damaged atoms;
- damage localization;
- post-peak load retention.

Do not automatically call work to failure "fracture toughness."

If a fracture-toughness-like quantity is reported, define it rigorously and state whether it is:
- a true fracture-mechanics quantity;
- or a simulation-derived proxy.

──────────────────────────────────────────────────────────────────────────────
DAMAGE AND CONNECTIVITY ANALYSIS
──────────────────────────────────────────────────────────────────────────────

The reactive potential does not require a permanent explicit bond table.

Do not use visualization bonds as the mechanical model.

For analysis, infer connectivity using a documented geometric or potential-consistent criterion.

Track persistent:

- lost neighbors;
- new neighbors;
- coordination changes;
- crack-surface formation;
- detached fragments;
- spatial location of damage events.

Define a quantitative damage-localization measure.

For example:

H = -Σ p_i log(p_i)

L = 1 - H/Hmax

where p_i describes the spatial distribution of irreversible damage.

Clearly define the binning and persistence rules.

──────────────────────────────────────────────────────────────────────────────
AUTONOMOUS SCIENTIFIC STUDY
──────────────────────────────────────────────────────────────────────────────

After the simulator and force field have passed validation, perform an actual computational discovery study.

Do not stop after creating the software.

Do not stop after generating structures.

Actually perform simulations and allow results to guide subsequent simulations.

Use approximately 80 primary simulations as an initial target, with additional runs where necessary to resolve important ambiguities.

Maintain a machine-readable experiment database containing every attempted structure and simulation.

──────────────────────────────────────────────────────────────────────────────
STAGE 1 — BASELINES
──────────────────────────────────────────────────────────────────────────────

Establish a small number of physically interpretable references.

At minimum include:

- pristine armchair graphene;
- pristine zigzag graphene;
- representative defected graphene;
- representative precracked graphene.

Add only the additional baseline architectures needed to interpret the image-derived design space.

──────────────────────────────────────────────────────────────────────────────
STAGE 2 — BROAD RECONNAISSANCE
──────────────────────────────────────────────────────────────────────────────

Explore the major architecture variables inferred from the images.

Ensure broad coverage rather than dense sampling of one small parameter region.

Hierarchy must receive meaningful reconnaissance coverage.

Include multiple hierarchy depths and multiple realizations of hierarchical organization rather than treating "hierarchical" as a binary label.

Identify:

- promising regimes;
- unexpectedly poor regimes;
- instabilities;
- apparent thresholds;
- surprising fracture mechanisms;
- variables that appear unimportant.

──────────────────────────────────────────────────────────────────────────────
STAGE 3 — FORM COMPETING HYPOTHESES
──────────────────────────────────────────────────────────────────────────────

After reconnaissance, formulate at least three competing mechanistic hypotheses.

The hypotheses must arise from the simulation observations rather than being copied from examples in this prompt.

For each hypothesis record:

- observation that motivated it;
- proposed mechanism;
- competing explanation;
- predicted observable if the mechanism is correct;
- predicted observable if the competing explanation is correct.

Prefer explanations based on physical mechanisms such as:

- local stress concentration;
- bond orientation;
- load-path redistribution;
- connectivity changes;
- geometric compliance;
- sequential recruitment;
- defect interactions;
- localization transitions.

Do not assume any of these mechanisms beforehand.

──────────────────────────────────────────────────────────────────────────────
STAGE 4 — DISCRIMINATING EXPERIMENTS
──────────────────────────────────────────────────────────────────────────────

Design simulations specifically to distinguish competing hypotheses.

Whenever possible create matched pairs in which only the scientifically relevant variable changes.

Examples of useful controlled comparisons include:

- same porosity, different architecture;
- same mass, different hierarchy;
- same hierarchy, different disorder;
- same topology, different anisotropy;
- same defect density, different organization.

Record predicted outcomes before running each discriminating experiment.

──────────────────────────────────────────────────────────────────────────────
STAGE 5 — DEEP MECHANISM STUDIES
──────────────────────────────────────────────────────────────────────────────

Once important variables have been identified, explore them systematically.

Search for:

- scaling laws;
- thresholds;
- non-monotonicity;
- regime transitions;
- parameter interactions;
- architecture-dependent mechanisms;
- failure-mode transitions;
- counterexamples.

Hierarchy should receive particular attention here if reconnaissance indicates that it affects mechanics.

Do not merely compare "hierarchical" versus "non-hierarchical."

Where possible identify how performance depends on:

- number of hierarchical levels;
- scale ratios;
- alignment between levels;
- connectivity across levels;
- organization of defects across levels;
- redundancy at each level.

──────────────────────────────────────────────────────────────────────────────
STAGE 6 — RANDOM-SEED STATISTICS
──────────────────────────────────────────────────────────────────────────────

For important stochastic or disordered structures use at least 3 independent seeds.

Prefer 5 seeds for conclusions where stochastic variability is substantial.

Report individual simulations as well as summary statistics.

──────────────────────────────────────────────────────────────────────────────
STAGE 7 — HOLDOUT PREDICTIONS
──────────────────────────────────────────────────────────────────────────────

Before running the final set of at least 10 unseen designs, record explicit predictions.

Predict:

- modulus;
- ultimate strength;
- failure strain;
- work to failure;
- damage localization;
- qualitative fracture mode;
- expected crack path or dominant damage mechanism.

Store predictions before the simulations begin.

──────────────────────────────────────────────────────────────────────────────
STAGE 8 — HOLDOUT EVALUATION
──────────────────────────────────────────────────────────────────────────────

Run the holdout simulations.

Compare predictions against observations.

Calculate quantitative prediction errors.

Identify:

- successful hypotheses;
- failed hypotheses;
- mechanisms requiring revision;
- unexpected regimes.

Do not retrospectively modify predictions.

──────────────────────────────────────────────────────────────────────────────
PARETO AND DESIGN ANALYSIS
──────────────────────────────────────────────────────────────────────────────

Do not optimize a single objective.

Identify Pareto tradeoffs among quantities such as:

- stiffness;
- strength;
- failure strain;
- work to failure;
- specific work;
- carbon mass;
- porosity;
- flaw tolerance;
- damage localization.

Preserve structurally diverse Pareto-optimal solutions.

Do not allow the optimization algorithm to collapse onto one architecture family merely because it dominates one scalar metric.

──────────────────────────────────────────────────────────────────────────────
ADAPTIVE EXPERIMENT SELECTION
──────────────────────────────────────────────────────────────────────────────

Use results from completed simulations to select informative new simulations.

Selection may consider:

- hypothesis discrimination;
- unexplored design regions;
- regime boundaries;
- unusual outliers;
- uncertainty;
- hierarchy-depth transitions;
- Pareto-front refinement;
- counterexamples.

Machine learning may assist experiment selection.

Machine learning must not replace the atomistic simulator or the mechanistic reasoning.

──────────────────────────────────────────────────────────────────────────────
EXPERIMENT RECORD
──────────────────────────────────────────────────────────────────────────────

For every experiment record:

- run ID;
- parent design ID;
- carbon architecture;
- complete geometry parameters;
- hierarchy descriptors;
- random seed;
- atom count;
- carbon mass;
- force field;
- force-field version;
- backend;
- device;
- precision;
- boundary conditions;
- minimization settings;
- strain increment;
- temperature if applicable;
- reason for selecting the experiment;
- hypothesis being tested;
- prediction if applicable;
- convergence status;
- structural viability;
- mechanical metrics;
- fracture mode;
- output paths;
- interpretation;
- subsequent decision.

Never silently remove anomalous simulations.

Classify anomalies as:

- geometry generation failure;
- structural instability;
- numerical failure;
- minimization failure;
- stochastic variation;
- force-field issue;
- genuine unexpected mechanical behavior.

──────────────────────────────────────────────────────────────────────────────
FRACTURE VISUALIZATION
──────────────────────────────────────────────────────────────────────────────

The visualization must be scientifically useful rather than decorative.

Allow visualization of:

- initial atomic structure;
- relaxed structure;
- deformed configurations;
- atomic displacement;
- local potential energy;
- virial stress;
- coordination;
- topology changes;
- crack initiation;
- crack propagation;
- detached fragments;
- accumulated damage.

Allow:

- top view;
- perspective view;
- orbit;
- zoom;
- pan;
- frame scrubbing;
- synchronized stress-strain display;
- playback.

──────────────────────────────────────────────────────────────────────────────
TOP-STRUCTURE FRACTURE PROGRESSION
──────────────────────────────────────────────────────────────────────────────

After completing the scientific analysis, automatically select approximately 6–10 structures that best illustrate the discoveries.

Do not simply select the strongest structures.

Choose structures that span important regimes such as:

- a reference state;
- high-performing designs;
- unusual Pareto designs;
- important hierarchical structures;
- localized fracture;
- progressive fracture;
- flaw-tolerant behavior;
- unexpected mechanisms;
- major counterexamples.

For every selected structure produce a detailed fracture-progression sequence.

Select frames based on physical events rather than only evenly spaced strain values.

At minimum extract:

1. relaxed structure before loading;
2. pre-damage configuration;
3. first irreversible bond/topology change;
4. peak-load configuration;
5. onset of major crack propagation or damage avalanche;
6. representative post-peak configuration;
7. final failed configuration.

Where progressive fracture contains several discrete damage events, include additional frames showing those events.

For every snapshot show, where useful:

- atomic structure;
- broken or changed connectivity;
- local energy;
- local stress;
- displacement field;
- damage location.

Pair the snapshots with the corresponding stress-strain curve and mark the strain of each displayed frame on the curve.

Provide a brief mechanistic interpretation explaining:

- where failure initiated;
- why it initiated there;
- how load paths changed;
- how damage propagated;
- whether damage localized or spread;
- whether hierarchy or another architecture feature influenced the progression;
- what distinguishes the structure from a relevant comparison design.

Provide direct before/after comparisons.

Also generate comparative panels for contrasting mechanisms, for example:

localized brittle failure
versus
distributed progressive failure

or:

non-hierarchical architecture
versus
hierarchical architecture

where those comparisons are supported by the results.

──────────────────────────────────────────────────────────────────────────────
FRACTURE MOVIES
──────────────────────────────────────────────────────────────────────────────

Generate MP4 movies for several scientifically important structures.

At minimum include:

- pristine graphene;
- representative precracked graphene;
- one important architecture discovered from the image-derived design space;
- one localized fracture example;
- one progressive fracture example;
- one major hierarchical design if hierarchy proves scientifically relevant;
- one high-performing or Pareto-optimal structure.

Movies should show:

- loading progression;
- atomic deformation;
- damage initiation;
- crack evolution;
- final fracture.

Where useful color by local energy, stress, displacement, or damage time.

Retain the raw trajectory used to create each movie.

Record the frame corresponding to:

- first damage;
- peak stress;
- major fracture event;
- final failure.

──────────────────────────────────────────────────────────────────────────────
PUBLICATION-QUALITY FIGURES
──────────────────────────────────────────────────────────────────────────────

Generate mandatory foundational figures showing:

1. reference images and extracted design principles;
2. translation from imagery to atomic carbon architecture;
3. explored design space;
4. force-field validation;
5. energy/force agreement with authoritative implementation;
6. bond-separation validation;
7. CPU/GPU precision where relevant;
8. computational performance;
9. representative stress-strain curves;
10. structural descriptor versus mechanical-property relationships;
11. hierarchy-depth and hierarchy-property relationships;
12. major mechanism-discrimination experiments;
13. localized versus progressive fracture;
14. Pareto fronts;
15. holdout prediction versus observation;
16. top-structure fracture-progression panels.

Generate additional figures based on the actual scientific discoveries rather than from a predetermined checklist.

Show individual observations.

Show variability and seed count where meaningful.

Every figure must have:

- labeled axes;
- units;
- readable legend;
- complete caption.

Export figures as:

- SVG;
- PDF;
- high-resolution PNG.

──────────────────────────────────────────────────────────────────────────────
APPLICATION
──────────────────────────────────────────────────────────────────────────────

Create a polished browser-based application with a local computational backend.

A suitable architecture is:

browser frontend
        ↓
local Python API
        ↓
ASE structure / experiment manager
        ↓
screened REBO2 engine
        ↓
MPS / CUDA / CPU where available
        ↓
trajectories and analysis
        ↓
interactive visualization

Display prominently:

FORCE ENGINE
- potential;
- implementation;
- device;
- precision;
- atom count;
- backend status.

If the force engine is unavailable, do not fabricate results.

Display actionable diagnostics.

Provide panels for:

- reference-image interpretation;
- geometry generation;
- simulation settings;
- physics/model description;
- validation;
- performance;
- results;
- top structures;
- fracture progression;
- experiment database.

──────────────────────────────────────────────────────────────────────────────
RESULTS AND EXPORT
──────────────────────────────────────────────────────────────────────────────

Allow export of:

- initial structure;
- relaxed structure;
- trajectories;
- final structure;
- ASE trajectory;
- XYZ;
- Extended XYZ;
- LAMMPS data;
- raw stress-strain data;
- CSV metrics;
- JSON metadata;
- per-atom quantities;
- PNG screenshots;
- SVG/PDF figures;
- selected fracture-frame panels;
- MP4 movies.

──────────────────────────────────────────────────────────────────────────────
SCIENTIFIC REPORT
──────────────────────────────────────────────────────────────────────────────

Write and compile a self-contained LaTeX scientific report.

Include:

- title;
- abstract;
- scientific question;
- interpretation of reference images;
- inferred design language;
- generated carbon architecture space;
- hierarchy design variables;
- force field;
- implementation;
- validation;
- bond-separation validation;
- numerical precision;
- computational performance;
- AQS protocol;
- experimental strategy;
- competing hypotheses;
- discriminating simulations;
- hierarchy studies;
- major mechanisms;
- rejected hypotheses;
- counterexamples;
- Pareto analysis;
- holdout predictions;
- holdout results;
- top-structure fracture progression;
- limitations;
- threats to validity;
- conclusions.

Include the complete experiment database as supplementary machine-readable data or an appendix where appropriate.

──────────────────────────────────────────────────────────────────────────────
CLAIM DISCIPLINE
──────────────────────────────────────────────────────────────────────────────

Clearly distinguish among:

- interpretation of reference images;
- generated structure;
- relaxed structure;
- directly calculated result;
- mechanistic interpretation;
- hypothesis;
- holdout prediction;
- rejected hypothesis;
- behavior of the chosen classical model;
- experimentally established behavior.

Do not describe simulations as physical experiments.

Do not claim that a computationally stable structure is synthesizable.

Do not call generated structures new allotropes without appropriate evidence.

Do not treat a work-to-failure integral as experimentally established fracture toughness.

──────────────────────────────────────────────────────────────────────────────
REPRODUCIBILITY
──────────────────────────────────────────────────────────────────────────────

Record:

- operating system;
- CPU;
- GPU;
- active device;
- precision;
- Python version;
- PyTorch version;
- ASE version;
- Atomistica version;
- force-field version;
- force-field source;
- git commit;
- random seeds;
- geometry parameters;
- hierarchy parameters;
- boundary conditions;
- numerical tolerances;
- strain increments;
- simulation dates.

Every experiment must be reproducible from its run ID.

──────────────────────────────────────────────────────────────────────────────
PROJECT STRUCTURE
──────────────────────────────────────────────────────────────────────────────

Organize the project clearly, for example:

carbon_discovery/
    app/
        frontend/
        backend/
    atomistics/
        structures/
        generators/
        descriptors/
    potentials/
        rebo2scr/
            reference/
            pytorch/
            tests/
    simulation/
        minimization/
        aqs/
        md/
        neighborlist/
    validation/
    experiments/
        database/
        predictions/
        holdouts/
    data/
        raw/
        processed/
    trajectories/
    analysis/
    figures/
    movies/
    final_designs/
    report/
    tests/
    scripts/
    README.md
    requirements.txt
    environment.yml
    manifest.json

──────────────────────────────────────────────────────────────────────────────
FINAL DELIVERABLES
──────────────────────────────────────────────────────────────────────────────

Deliver:

- working browser application;
- complete source code;
- screened REBO2 integration;
- GPU implementation if successfully validated;
- ASE integration;
- automated tests;
- validation data;
- bond-separation validation;
- CPU/GPU precision comparison;
- complete experiment database;
- all generated structures;
- all relaxed structures;
- raw trajectories;
- processed data;
- holdout predictions;
- holdout outcomes;
- publication-quality figures;
- fracture-progression snapshot panels;
- MP4 movies;
- selected scientifically informative structures;
- report.tex;
- bibliography;
- compiled report.pdf;
- environment specification;
- launch scripts;
- manifest;
- file checksums.

Package all deliverables into one ZIP archive.

──────────────────────────────────────────────────────────────────────────────
FINAL QUALITY CONTROL
──────────────────────────────────────────────────────────────────────────────

Before completion:

1. launch the application from a clean environment;
2. verify force-field availability;
3. verify actual device use;
4. run all validation tests;
5. verify bond-separation behavior;
6. compare accelerated and reference calculations;
7. inspect stress-strain curves;
8. inspect fracture trajectories;
9. verify that plots derive from actual simulation data;
10. verify experiment reproducibility;
11. verify holdout predictions were written before simulations;
12. verify selected fracture frames correspond to the stated physical events;
13. inspect all fracture movies;
14. compile the LaTeX report;
15. visually inspect every report page;
16. repair clipped plots, overlaps, unreadable labels, broken references, and missing figures;
17. verify that the final ZIP contains all required artifacts.

──────────────────────────────────────────────────────────────────────────────
COMPLETION CRITERION
──────────────────────────────────────────────────────────────────────────────

The task is complete only when:

1. explicit atomistic carbon structures have been generated;
2. the structures are represented consistently through ASE;
3. screened REBO2 is functioning as the production force field;
4. the implementation has been validated against an authoritative reference;
5. bond-separation behavior has been explicitly validated;
6. numerical precision has been quantified;
7. structural relaxation validation passes;
8. mechanical validation passes;
9. strain-step and size convergence have been examined;
10. the autonomous discovery study has actually been executed;
11. hierarchy has been explored systematically as a genuine design variable;
12. competing hypotheses have been generated from the simulation observations;
13. discriminating experiments have been run;
14. stochastic structures have been replicated;
15. at least 10 holdout predictions have been recorded before simulation;
16. the holdout designs have been simulated and evaluated;
17. Pareto-optimal architectures have been identified;
18. top scientifically informative structures have been selected;
19. before/during/after fracture snapshots have been generated for those structures;
20. several scientifically useful fracture movies have been generated;
21. every figure comes from actual simulation data;
22. raw trajectories and data have been retained;
23. the complete scientific report has been compiled and visually inspected;
24. source, data, structures, figures, movies, metadata, and report are packaged into one reproducible ZIP archive.

Do not stop at an attractive web interface.

Do not substitute a spring model.

Do not substitute a machine-learned potential.

Do not fabricate simulation results.

Do not silently use another force field.

Do not silently move calculations from GPU to CPU.

Do not sacrifice scientific correctness for computational speed.

The /goal:

Build, validate, and use a complete generative atomistic scientific-discovery platform for graphene and related two-dimensional carbon architectures.

Use published screened REBO2 as the primary production force field.

Use ASE as the canonical atomic representation.

Use GPU acceleration on Apple MPS and NVIDIA CUDA when an exact, validated implementation is feasible, while retaining CPU/reference execution.

Infer carbon design principles from the supplied images rather than imposing a predefined structural interpretation.

Explore the resulting architecture space deeply, with particular emphasis on hierarchy as a controlled multivariable design axis, together with disorder, anisotropy, connectivity, redundancy, porosity, defects, cracks, and load-path organization.

Use autonomous hypothesis formation and discriminating simulations to determine which architectural variables actually govern fracture.

Make quantitative predictions on unseen structures before simulating them.

Identify mechanisms, regime transitions, scaling relationships, tradeoffs, counterexamples, and Pareto-optimal designs.

For the most scientifically informative structures, explicitly analyze the evolution from the intact structure through first damage, peak load, crack propagation, progressive damage where applicable, and final fracture using event-selected snapshots and fracture movies.

Deliver the complete validated simulator, source, experiment database, atomic structures, trajectories, figures, fracture visualizations, movies, holdout analysis, selected designs, and compiled scientific report as one reproducible project.
