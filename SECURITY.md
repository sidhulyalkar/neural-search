# Security Policy

## Supported versions

Neural Search is currently an alpha research project. Security fixes are applied to the latest `main` branch rather than maintained release branches.

## Reporting a vulnerability

Please do not publish credentials, private data, exploitable endpoints, or proof-of-concept attacks in a public issue.

Use GitHub's private vulnerability reporting feature for this repository when available. If private reporting is not available, contact the repository maintainer privately through their GitHub profile before disclosing details publicly.

Include:

- the affected component and commit or version;
- reproduction steps;
- realistic impact;
- whether credentials, private datasets, or remote execution are involved;
- a suggested mitigation, if known.

## Sensitive data and credentials

Neural Search integrates with external scientific repositories and can optionally use API keys or database credentials. Contributors should:

- keep secrets in local environment variables or untracked `.env` files;
- never commit tokens, credentials, private dataset contents, or signed URLs;
- treat downloaded participant-level data according to its source license, consent, and governance requirements;
- avoid logging raw secrets or sensitive participant metadata.

The example environment file must contain placeholders only.
