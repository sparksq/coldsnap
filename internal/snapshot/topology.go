// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package snapshot

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"

	"github.com/sparksq/coldsnap/internal/canonicaljson"
)

const (
	UnitOwnerPrefix    = "unit/"
	WorkerOwnerPrefix  = "worker/"
	ServiceOwnerPrefix = "service/"
	GroupOwnerPrefix   = "group/"
)

// ExecutionGraph describes the captured process topology without assuming
// that engine parallelism is a Cartesian product. Launch units are OS/container
// process-tree boundaries. Workers are accelerator-owning engine processes.
// Ordered groups carry every rank namespace used by the engine.
type ExecutionGraph struct {
	Workers  []Worker        `json:"workers"`
	Groups   []ProcessGroup  `json:"groups"`
	Services []ServiceDomain `json:"services"`
	Adapter  AdapterTopology `json:"adapter"`
}

// Worker is a portable process slot inside a launch unit. DeviceSlots index
// the owning LaunchUnit.Devices list; physical ordinals and UUIDs remain
// placement data rather than artifact identity.
type Worker struct {
	ID          string `json:"id"`
	Unit        string `json:"unit"`
	Service     string `json:"service"`
	ProcessSlot int    `json:"process_slot"`
	DeviceSlots []int  `json:"device_slots"`
}

// ProcessGroup is an ordered rank namespace. Kind is adapter-owned and
// namespaced (for example vllm:tensor or sglang:dp-attention); ColdSnap does
// not enumerate parallelism modes.
type ProcessGroup struct {
	ID      string   `json:"id"`
	Kind    string   `json:"kind"`
	Service string   `json:"service"`
	Members []string `json:"members"`
}

// ServiceDomain separates independently addressable engines or cooperating
// roles such as data-parallel replicas and prefill/decode services.
type ServiceDomain struct {
	ID      string   `json:"id"`
	Role    string   `json:"role"`
	Workers []string `json:"workers"`
}

// AdapterTopology is deliberately opaque to core ColdSnap. The engine adapter
// owns Schema and Payload; Digest makes the observed topology a capture
// compatibility boundary without requiring a core format change for new
// parallel modes.
type AdapterTopology struct {
	Schema  string          `json:"schema"`
	Digest  string          `json:"digest"`
	Payload json.RawMessage `json:"payload"`
}

// RuntimeProviders binds adapter-defined runtime state to a launch unit. The
// core artifact format deliberately treats the payload as opaque: the adapter
// owns its schema and compatibility rules while ColdSnap owns placement,
// canonical encoding, and digest integrity.
type RuntimeProviders struct {
	Bindings []RuntimeProviderBinding `json:"bindings,omitempty"`
}

type RuntimeProviderBinding struct {
	Owner   string          `json:"owner"`
	Kind    string          `json:"kind"`
	Schema  string          `json:"schema"`
	Digest  string          `json:"digest"`
	Payload json.RawMessage `json:"payload"`
}

func NewRuntimeProviderBinding(owner, kind, name string, payload any) (RuntimeProviderBinding, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return RuntimeProviderBinding{}, fmt.Errorf("encode runtime provider: %w", err)
	}
	digest, err := canonicaljson.CanonicalSHA256(data)
	if err != nil {
		return RuntimeProviderBinding{}, fmt.Errorf("canonicalize runtime provider: %w", err)
	}
	result := RuntimeProviderBinding{
		Owner: owner, Kind: kind, Schema: name, Digest: "sha256:" + digest, Payload: data,
	}
	if err := result.validate(); err != nil {
		return RuntimeProviderBinding{}, err
	}
	return result, nil
}

func (providers RuntimeProviders) Validate(launch LaunchSpec) error {
	units := make(map[string]bool, len(launch.Units))
	for _, unit := range launch.Units {
		units[UnitOwner(unit.ID)] = true
	}
	seen := make(map[string]bool, len(providers.Bindings))
	for _, binding := range providers.Bindings {
		if !units[binding.Owner] {
			return fmt.Errorf("runtime provider owner %q is not a launch unit", binding.Owner)
		}
		key := binding.Owner + "\x00" + binding.Kind
		if seen[key] {
			return fmt.Errorf("runtime provider %s for %s is duplicated", binding.Kind, binding.Owner)
		}
		seen[key] = true
		if err := binding.validate(); err != nil {
			return fmt.Errorf("runtime provider %s for %s: %w", binding.Kind, binding.Owner, err)
		}
	}
	return nil
}

func (binding RuntimeProviderBinding) validate() error {
	if !strings.HasPrefix(binding.Owner, UnitOwnerPrefix) ||
		!qualifiedName(binding.Kind) || !qualifiedName(binding.Schema) ||
		!digestPattern.MatchString(binding.Digest) || len(binding.Payload) == 0 {
		return errors.New("runtime provider identity is incomplete")
	}
	var payload any
	decoder := json.NewDecoder(bytes.NewReader(binding.Payload))
	decoder.UseNumber()
	if err := decoder.Decode(&payload); err != nil {
		return errors.New("runtime provider payload is invalid JSON")
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return errors.New("runtime provider payload is invalid JSON")
	}
	if _, ok := payload.(map[string]any); !ok {
		return errors.New("runtime provider payload must be an object")
	}
	digest, err := canonicaljson.CanonicalSHA256(binding.Payload)
	if err != nil {
		return fmt.Errorf("canonicalize runtime provider: %w", err)
	}
	if binding.Digest != "sha256:"+digest {
		return errors.New("runtime provider payload digest mismatch")
	}
	return nil
}

func NewAdapterTopology(name string, payload any) (AdapterTopology, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return AdapterTopology{}, fmt.Errorf("encode adapter topology: %w", err)
	}
	digest, err := canonicaljson.CanonicalSHA256(data)
	if err != nil {
		return AdapterTopology{}, fmt.Errorf("canonicalize adapter topology: %w", err)
	}
	result := AdapterTopology{Schema: name, Digest: "sha256:" + digest, Payload: data}
	if err := result.Validate(); err != nil {
		return AdapterTopology{}, err
	}
	return result, nil
}

func (graph ExecutionGraph) Validate(units []LaunchUnit) error {
	if len(units) == 0 || len(graph.Workers) == 0 || len(graph.Groups) == 0 || len(graph.Services) == 0 {
		return errors.New("execution graph requires units, workers, groups, and services")
	}
	unitByID := make(map[string]LaunchUnit, len(units))
	for _, unit := range units {
		if !idPattern.MatchString(unit.ID) || unitByID[unit.ID].ID != "" {
			return errors.New("execution graph contains an invalid or duplicate unit ID")
		}
		unitByID[unit.ID] = unit
	}
	serviceByID := make(map[string]ServiceDomain, len(graph.Services))
	for _, service := range graph.Services {
		if !idPattern.MatchString(service.ID) || !qualifiedName(service.Role) || serviceByID[service.ID].ID != "" || len(service.Workers) == 0 {
			return errors.New("execution graph contains an invalid service")
		}
		serviceByID[service.ID] = service
	}
	workerByID := make(map[string]Worker, len(graph.Workers))
	unitSlots := make(map[string]map[int]bool, len(units))
	unitDevices := make(map[string]map[int]bool, len(units))
	for _, worker := range graph.Workers {
		unit, unitOK := unitByID[worker.Unit]
		_, serviceOK := serviceByID[worker.Service]
		if !idPattern.MatchString(worker.ID) || workerByID[worker.ID].ID != "" || !unitOK || !serviceOK || worker.ProcessSlot < 0 || len(worker.DeviceSlots) == 0 {
			return errors.New("execution graph contains an invalid worker")
		}
		if unitSlots[worker.Unit] == nil {
			unitSlots[worker.Unit] = make(map[int]bool)
			unitDevices[worker.Unit] = make(map[int]bool)
		}
		if unitSlots[worker.Unit][worker.ProcessSlot] {
			return fmt.Errorf("unit %s contains duplicate process slot %d", worker.Unit, worker.ProcessSlot)
		}
		unitSlots[worker.Unit][worker.ProcessSlot] = true
		seenDeviceSlots := make(map[int]bool)
		for _, slot := range worker.DeviceSlots {
			if slot < 0 || slot >= len(unit.Devices) || seenDeviceSlots[slot] || unitDevices[worker.Unit][slot] {
				return fmt.Errorf("worker %s contains an invalid or shared device slot", worker.ID)
			}
			seenDeviceSlots[slot] = true
			unitDevices[worker.Unit][slot] = true
		}
		workerByID[worker.ID] = worker
	}
	for _, unit := range units {
		if len(unitSlots[unit.ID]) == 0 {
			return fmt.Errorf("launch unit %s has no workers", unit.ID)
		}
	}
	serviceMembers := make(map[string]map[string]bool, len(graph.Services))
	for _, service := range graph.Services {
		members := make(map[string]bool, len(service.Workers))
		for _, workerID := range service.Workers {
			worker, ok := workerByID[workerID]
			if !ok || worker.Service != service.ID || members[workerID] {
				return fmt.Errorf("service %s contains an invalid worker membership", service.ID)
			}
			members[workerID] = true
		}
		serviceMembers[service.ID] = members
	}
	for _, worker := range graph.Workers {
		if !serviceMembers[worker.Service][worker.ID] {
			return fmt.Errorf("worker %s is absent from service %s", worker.ID, worker.Service)
		}
	}
	groupIDs := make(map[string]bool, len(graph.Groups))
	groupedWorkers := make(map[string]bool, len(graph.Workers))
	worldMemberships := make(map[string]int, len(graph.Workers))
	for _, group := range graph.Groups {
		if !idPattern.MatchString(group.ID) || !qualifiedName(group.Kind) || groupIDs[group.ID] || len(group.Members) == 0 {
			return errors.New("execution graph contains an invalid process group")
		}
		if _, ok := serviceByID[group.Service]; !ok {
			return fmt.Errorf("process group %s references an unknown service", group.ID)
		}
		groupIDs[group.ID] = true
		members := make(map[string]bool, len(group.Members))
		for _, workerID := range group.Members {
			worker, ok := workerByID[workerID]
			if !ok || worker.Service != group.Service || members[workerID] {
				return fmt.Errorf("process group %s contains an invalid worker membership", group.ID)
			}
			members[workerID] = true
			groupedWorkers[workerID] = true
			if strings.HasSuffix(group.Kind, ":world") {
				worldMemberships[workerID]++
			}
		}
	}
	for _, worker := range graph.Workers {
		if !groupedWorkers[worker.ID] {
			return fmt.Errorf("worker %s belongs to no process group", worker.ID)
		}
		if worldMemberships[worker.ID] != 1 {
			return fmt.Errorf(
				"worker %s must belong to exactly one engine world group",
				worker.ID,
			)
		}
	}
	return graph.Adapter.Validate()
}

func (topology AdapterTopology) Validate() error {
	if !qualifiedName(topology.Schema) || !digestPattern.MatchString(topology.Digest) || len(topology.Payload) == 0 {
		return errors.New("adapter topology identity is incomplete")
	}
	var payload any
	decoder := json.NewDecoder(bytes.NewReader(topology.Payload))
	decoder.UseNumber()
	if err := decoder.Decode(&payload); err != nil {
		return errors.New("adapter topology payload is invalid JSON")
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return errors.New("adapter topology payload is invalid JSON")
	}
	if _, ok := payload.(map[string]any); !ok {
		return errors.New("adapter topology payload must be an object")
	}
	digest, err := canonicaljson.CanonicalSHA256(topology.Payload)
	if err != nil {
		return fmt.Errorf("canonicalize adapter topology: %w", err)
	}
	if topology.Digest != "sha256:"+digest {
		return errors.New("adapter topology payload digest mismatch")
	}
	return nil
}

func qualifiedName(value string) bool {
	if value == "" || strings.ContainsAny(value, " \t\r\n\x00") {
		return false
	}
	separator := strings.IndexByte(value, ':')
	return separator > 0 && separator < len(value)-1 && idPattern.MatchString(value[:separator]) && idPattern.MatchString(value[separator+1:])
}

func (graph ExecutionGraph) UnitWorkers(unitID string) []Worker {
	workers := make([]Worker, 0)
	for _, worker := range graph.Workers {
		if worker.Unit == unitID {
			workers = append(workers, worker)
		}
	}
	slices.SortFunc(workers, func(left, right Worker) int { return left.ProcessSlot - right.ProcessSlot })
	return workers
}

func (graph ExecutionGraph) Worker(id string) (Worker, bool) {
	for _, worker := range graph.Workers {
		if worker.ID == id {
			return worker, true
		}
	}
	return Worker{}, false
}

func UnitOwner(id string) string    { return UnitOwnerPrefix + id }
func WorkerOwner(id string) string  { return WorkerOwnerPrefix + id }
func ServiceOwner(id string) string { return ServiceOwnerPrefix + id }
func GroupOwner(id string) string   { return GroupOwnerPrefix + id }

func (launch LaunchSpec) validOwners() map[string]bool {
	owners := make(map[string]bool, len(launch.Units)+len(launch.Execution.Workers)+len(launch.Execution.Services)+len(launch.Execution.Groups))
	for _, unit := range launch.Units {
		owners[UnitOwner(unit.ID)] = true
	}
	for _, worker := range launch.Execution.Workers {
		owners[WorkerOwner(worker.ID)] = true
	}
	for _, service := range launch.Execution.Services {
		owners[ServiceOwner(service.ID)] = true
	}
	for _, group := range launch.Execution.Groups {
		owners[GroupOwner(group.ID)] = true
	}
	return owners
}
