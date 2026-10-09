# Security policy

## Supported versions

Dedrift is pre-1.0 alpha software. Only the newest published release receives
security fixes. It is not approved here for storing secrets, regulated data,
or unredacted sensitive prompts and outputs.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Email
`support@dedrift.ai` with `SECURITY` in the subject and include:

- affected version or commit;
- reproduction steps and impact;
- whether data exposure or integrity loss is ongoing;
- a safe way to contact you.

Receipt should be acknowledged within three business days. No remediation SLA
is promised while the project is alpha. A coordinated disclosure date will be
agreed when a fix is available.

## Current data-handling warning

Dedrift stores interaction inputs, outputs, structured data, errors, and agent
configuration locally in plaintext JSONL and SQLite, with an optional local
embedding cache. Filesystem permissions are hardened, but the project does not
yet provide application-level encryption, redaction, retention, RBAC, or secure
deletion. Apply redaction before logging and protect the complete `.dedrift`
directory as sensitive data.
