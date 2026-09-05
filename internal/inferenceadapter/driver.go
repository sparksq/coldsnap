// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"
	"fmt"

	"github.com/sparksq/coldsnap/internal/snapshot"
	"github.com/sparksq/coldsnap/internal/snapshotdriver"
)

// processDriver is the engine-adapter seam for process/CUDA snapshot
// implementations. Artifact, topology, capsule, cache, and weight-provider
// mechanics remain shared by the adapter.
type processDriver interface {
	Contract() snapshotdriver.Contract
	Capture(context.Context, Adapter, snapshot.Request) error
	Restore(context.Context, Adapter, snapshot.Request) error
	PrepareRestore(context.Context, Adapter, snapshot.Request) (restorePreparation, error)
}

type n610ProcessDriver struct {
	contract snapshotdriver.Contract
}

func (driver n610ProcessDriver) Contract() snapshotdriver.Contract { return driver.contract }

func (n610ProcessDriver) Capture(ctx context.Context, adapter Adapter, request snapshot.Request) error {
	return adapter.captureN610(ctx, request)
}

func (n610ProcessDriver) Restore(ctx context.Context, adapter Adapter, request snapshot.Request) error {
	return adapter.restore(ctx, request)
}

type n580ProcessDriver struct {
	contract snapshotdriver.Contract
}

func (driver n580ProcessDriver) Contract() snapshotdriver.Contract { return driver.contract }

func (n580ProcessDriver) Capture(ctx context.Context, adapter Adapter, request snapshot.Request) error {
	return adapter.captureN580(ctx, request)
}

func (n580ProcessDriver) Restore(ctx context.Context, adapter Adapter, request snapshot.Request) error {
	return adapter.restore(ctx, request)
}

func (n580ProcessDriver) PrepareRestore(
	ctx context.Context, adapter Adapter, request snapshot.Request,
) (restorePreparation, error) {
	return adapter.prepareRestore(ctx, request)
}

func (n610ProcessDriver) PrepareRestore(
	ctx context.Context, adapter Adapter, request snapshot.Request,
) (restorePreparation, error) {
	return adapter.prepareRestore(ctx, request)
}

func resolveProcessDriver(selection snapshotdriver.Selection) (processDriver, error) {
	contract, err := snapshotdriver.Resolve(selection)
	if err != nil {
		return nil, err
	}
	switch contract.ID {
	case snapshotdriver.N610:
		return n610ProcessDriver{contract: contract}, nil
	case snapshotdriver.N580:
		return n580ProcessDriver{contract: contract}, nil
	default:
		return nil, fmt.Errorf("snapshot driver %s has no inference-engine implementation", contract.ID)
	}
}

func validateArtifactDriver(request snapshot.Request, artifact snapshot.Artifact) error {
	contract, err := snapshotdriver.Resolve(request.Driver)
	if err != nil {
		return err
	}
	if artifact.Driver.ID != contract.ID || artifact.Driver.ABI != contract.ABI {
		return fmt.Errorf(
			"artifact snapshot driver %s ABI %d does not match requested %s ABI %d",
			artifact.Driver.ID, artifact.Driver.ABI, contract.ID, contract.ABI,
		)
	}
	return nil
}
