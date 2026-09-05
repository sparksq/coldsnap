<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Benchmarks

Only maintained measurement and qualification harnesses live here. See the
[harness inventory and prerequisites](harnesses/README.md).

Store results, logs, rendered recipes, host inventories, and historical
experiments outside the checkout, for example in a private benchmark store.
They can contain machine addresses, model-cache paths, and runtime environment
data. Do not commit them or package them into runtime images.

For an auditable run, retain the source commit, engine/image and model digests,
driver and hardware inventory, topology, requested policy, cache state,
validation outcome, and exact timing boundary with the private result.
Compare matched vanilla, native, and recovery cases. Capture, artifact staging,
and full verification are separate from Docker-start-to-first-token timing;
report them separately when measuring end-to-end user latency.

Runtime compatibility is defined by source contracts and release-owned NCCL
admission records, not by the presence of historical benchmark reports.
