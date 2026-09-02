# Security policy

NeuralRenderKit is pre-release software. Until a stable release exists, only the latest `main` revision receives security fixes.

## Reporting

Once the public repository enables GitHub private vulnerability reporting, use that channel for suspected vulnerabilities. Do not attach proprietary model files, credentials, private captures, personal data, or exploit payloads to a public issue. Include the affected revision, macOS and hardware version, a minimal synthetic package when possible, and the observed versus expected behavior.

## Trust boundary

Model packages are untrusted data. The runtime validates the manifest size and schema, requires a single safe relative weight filename, verifies SHA-256 before MLX parsing, and checks every declared weight name, shape, and dtype. It never loads executable code from a package.

These checks do not establish package authorship, model safety, or license rights. SHA-256 detects a mismatch against the manifest but is not a signature. A syntactically valid large or adversarial model may still consume substantial memory or exercise bugs in upstream parsers and GPU drivers. Hosts should accept packages only from trusted sources, enforce their own storage/resource policy, and isolate high-risk workloads where appropriate.

The public repository intentionally excludes proprietary binaries and weights, executable artifacts, private oracle data, and machine-specific paths. Run `scripts/audit-public-tree.sh .` before publication.

## Supported security scope

Security reports are welcome for package path traversal, integrity-check bypass, unsafe size/shape arithmetic, memory-safety issues, unintended executable loading, sensitive-data disclosure, or a reproducible denial of service that crosses the documented host resource boundary.
