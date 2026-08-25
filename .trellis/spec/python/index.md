# Python

## Applicability and authority

This layer applies to Python source, tests, reusable configs, and thin experiment entry points. Cross-module rules live here; module-local architecture stays in the owning `src/nimloth/**/README.md`.

## Pre-Development Checklist

- Read the [structure/module index](structure-and-module-indexes.md) and every owning module README.
- Search adjacent source and tests for an existing implementation before adding code.
- Read [configuration and interface boundaries](configuration-and-interfaces.md).
- Identify mirrored test directories and focused/full-scope checks before editing.
- Confirm experiment changes also load the experiments layer and require an experiment task/launch gate.

## Quality Check

- Code is in the correct module; ownership and dependency direction remain clear.
- New/changed module boundaries are indexed in the module README.
- Runtime code consumes validated typed config rather than parsing YAML or inventing defaults.
- Focused tests, affected-package tests, compile/type/lint checks, and semantic invariants pass at an appropriate scope.
- Errors are visible; no temporary stand-in, mock result, silent fallback, or weakened assertion masks the requested behavior.

## Topic specs

- [Source structure and module README index](structure-and-module-indexes.md)
- [Configuration, interfaces, and cross-module boundaries](configuration-and-interfaces.md)
- [Quality and testing](quality-and-testing.md)

## Source indexes

- [`src/nimloth/README.md`](../../../src/nimloth/README.md)
- [`src/nimloth/training/README.md`](../../../src/nimloth/training/README.md)
- [`tests/`](../../../tests/)
- [`configs/`](../../../configs/)
