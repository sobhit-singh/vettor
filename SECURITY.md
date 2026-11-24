# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately to the maintainer instead
of opening a public issue. Include the affected version, reproduction steps,
and the potential impact. You should receive an acknowledgement once the
report has been reviewed.

## Supported versions

Security fixes are applied to the latest released version.

## Scope

Vettor can be pointed at arbitrary web applications and can send configured
authentication headers, cookies, and storage state to those targets. Treat
configuration files and CLI arguments that contain credentials as sensitive:
do not commit them, and prefer environment variables or runtime flags.
