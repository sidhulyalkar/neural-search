# Task 28: Species and Model-Organism Intelligence

Status: first implementation slice completed.

This task introduces canonical species and broader animal-type understanding for neuroscience search and graph explanations.

Implemented behavior:

- Canonical species profiles for human, mouse, rat, macaque, marmoset, zebrafish, drosophila, C. elegans, ferret, songbird, and mixed-species records.
- Query expansion for aliases such as `NHP`, `non-human primate`, `rodent`, `mouse model`, `larval zebrafish`, `fruit fly`, and `C. elegans`.
- Hard-negative filtering for broad animal types, including queries such as `without rodents`.
- `human only` style queries become hard-negative species constraints for non-human species.
- Graph builds now attach species nodes to taxon groups, animal types, and model-organism roles.
- Graph search features can expose species context and broader taxon matches.

Next expansion should add homologous brain-region mappings and species-specific caveats for cross-species association paths.
