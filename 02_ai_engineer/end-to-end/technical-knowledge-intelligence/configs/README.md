# Configuration layout

`config.yaml` remains the single runtime source of truth because the application is a local-first monorepo and splitting every setting into multiple files would add complexity without operational benefit.

This directory contains task-specific reference profiles used for evaluation/deployment documentation. Secrets never belong here; use `.env` locally or a platform secret manager in deployment.
