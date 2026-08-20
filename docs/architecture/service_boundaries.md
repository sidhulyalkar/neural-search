# API and Service Boundaries

Neural Search grew organically from a demo API into a broad research application. `apps/api/main.py` now contains substantial transport, startup, cache, search adaptation, feedback, literature, QA, report, and export logic. Rewriting that module in one pass would create a large regression surface with little scientific benefit.

This document defines the incremental migration boundary instead.

## Target layering

```text
apps/api/
  application.py       composition root
  *_router.py          HTTP transport only
          ↓
neural_search/services/
  application services / use cases
          ↓
neural_search/search/
neural_search/graph/
neural_search/literature/
neural_search/evaluation/
  scientific/domain logic
          ↓
neural_search/runtime/
neural_search/ingestion/
  artifact and infrastructure boundaries
```

The important dependency rule is **downward only**. Scientific domain modules must not import FastAPI request/response types or frontend-specific state.

## Composition root

`apps/api/application.py` is the canonical application composition point for new API infrastructure.

The legacy `apps.api.main:app` remains available for backward compatibility while routes are migrated. New cross-cutting or infrastructure routers should be registered at the composition root rather than adding more unrelated responsibilities to `main.py`.

Containers should launch the composition root:

```bash
uvicorn apps.api.application:app --host 0.0.0.0 --port 8000
```

Local developer commands should use the same entry point so Docker, CI, and local development do not expose different API surfaces.

## Router responsibilities

A router may:

- validate HTTP input;
- map request models to an application-service call;
- map domain/service results to HTTP responses;
- translate expected domain errors into HTTP status codes;
- enforce transport-level authorization when authentication is introduced.

A router should not:

- scan large artifact directories;
- implement retrieval/ranking logic;
- mutate qrels or scientific evidence tiers;
- contain corpus normalization algorithms;
- own long-lived caches whose semantics matter outside HTTP;
- silently fall back between scientifically different artifact revisions.

## Application services

As legacy endpoints are touched, extract their orchestration into small application services. A useful service should represent a user goal rather than a generic utility, for example:

- `SearchDatasets`
- `SearchLiterature`
- `GenerateReuseCard`
- `GenerateStarterNotebook`
- `RecordRetrievalFeedback`
- `InspectRuntimeReadiness`

The service layer is where a request can coordinate several domain components while remaining independent of FastAPI. This makes the same operation callable from the CLI, tests, notebooks, agents, and future workers without copying route logic.

## Artifact access

First-class artifacts should be resolved through an explicit contract rather than adding new module-level path constants throughout routers. The runtime registry already establishes IDs, lifecycle kinds, and profile relationships. A future `ArtifactStore` abstraction can add versioning and remote acquisition without changing every endpoint.

Until that exists, new code should at minimum:

1. register durable artifacts in `neural_search/runtime/catalog.py`;
2. surface missing generated assets as capability/readiness information;
3. avoid treating an absent production artifact as a surprising exception in profiles where it is optional.

## Migration strategy for `apps/api/main.py`

Do not split the file by line count alone. Migrate by cohesive user capability, preserving tests at each step.

Recommended sequence:

1. runtime/profile inspection, now separated into `runtime_router.py`;
2. search session and user-feedback endpoints;
3. literature search and evidence endpoints;
4. dataset card/notebook/export endpoints;
5. core dataset search adapter;
6. QA/admin mutation surfaces.

For each migration:

- add a service-level test before moving behavior;
- keep the HTTP contract stable unless the change is intentional and documented;
- run retrieval benchmarks for any change that can alter ranking or match explanations;
- remove legacy helpers only after all callers have moved.

## Why this architecture serves external users

The service boundary is not merely code cleanup. It creates three practical benefits:

- labs can import scientific functionality without running a web server;
- workers/agents can invoke the same use cases as the frontend without reimplementing route code;
- deployments can swap artifact storage, queues, or databases without coupling those choices to retrieval science.

That separation is particularly important for Neural Search because its long-term value is not only a website. The reusable core is an experiment-aware discovery and reanalysis substrate that should be callable from interactive tools, automated agents, notebooks, and reproducible pipelines.
