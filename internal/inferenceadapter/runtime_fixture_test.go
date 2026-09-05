// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package inferenceadapter

import (
	"context"

	"github.com/sparksq/coldsnap/internal/runtimetest"
	"github.com/sparksq/coldsnap/internal/snapshot"
)

func fixtureAdapter(adapter Adapter) Adapter {
	adapter.Runtime = runtimetest.NewDockerRuntime(adapter.Remote, "docker")
	return adapter
}
func rankCommandFixture(adapter Adapter,
	ctx context.Context,
	request snapshot.Request,
	unit snapshot.LaunchUnit,
	image, name, operation, namespace, captureID, artifactRoot, endpointPath, provider string,
	modelPayloads []snapshot.PreparedPayload,
	identityImage string,
	tcpAddressMap []string,
	tcpPortShift uint16,
	materialization *materializationMount,
) ([]string, error) {
	spec, err := adapter.rankWorkloadSpec(ctx, request, unit, image, name, operation, namespace, captureID, artifactRoot, endpointPath, provider, modelPayloads, identityImage, tcpAddressMap, tcpPortShift, materialization)
	if err != nil {
		return nil, err
	}
	return runtimetest.DockerRunArguments("docker", *spec)
}
