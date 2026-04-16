# Security Policy

VICTOR is a proprietary product of MASSIVEMAGNETICS.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability in VICCTORIAN, **please do not open a
public GitHub issue**. Instead:

1. **Email** the maintainers directly via GitHub private messaging, or
2. Use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
   feature (Security → Report a vulnerability).

Please include:

* A description of the vulnerability and its impact
* Steps to reproduce
* Any suggested mitigations

We aim to acknowledge reports within **48 hours** and provide a fix or mitigation
within **14 days** for confirmed vulnerabilities.

## Scope

* The `victor` Python package (all modules under `victor/`)
* The CLI entry point (`vicctorian` / `victor`)
* Configuration loading (`config.toml`, env vars)

## Out of Scope

* Third-party dependencies (report those to the upstream project)
* Issues in your own configuration or deployment environment
