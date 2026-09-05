// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-FileCopyrightText: 2026 Fox Engine Ltd
// SPDX-License-Identifier: AGPL-3.0-only

package ncclprovider

import (
	"bytes"
	"debug/elf"
	"encoding/hex"
	"errors"
	"fmt"
)

type elfIdentity struct {
	Architecture string
	BuildID      string
	SONAME       string
}

func inspectELF(path string) (elfIdentity, error) {
	file, err := elf.Open(path)
	if err != nil {
		return elfIdentity{}, err
	}
	defer file.Close()
	architecture := ""
	switch file.Machine {
	case elf.EM_AARCH64:
		architecture = "aarch64"
	case elf.EM_X86_64:
		architecture = "x86_64"
	default:
		return elfIdentity{}, fmt.Errorf("unsupported ELF machine %s", file.Machine)
	}
	sonames, err := file.DynString(elf.DT_SONAME)
	if err != nil {
		return elfIdentity{}, fmt.Errorf("read ELF SONAME: %w", err)
	}
	if len(sonames) != 1 || sonames[0] == "" {
		return elfIdentity{}, errors.New("ELF must contain exactly one SONAME")
	}
	buildID, err := gnuBuildID(file)
	if err != nil {
		return elfIdentity{}, err
	}
	return elfIdentity{Architecture: architecture, BuildID: buildID, SONAME: sonames[0]}, nil
}

func gnuBuildID(file *elf.File) (string, error) {
	section := file.Section(".note.gnu.build-id")
	if section == nil {
		return "", errors.New("ELF lacks .note.gnu.build-id")
	}
	data, err := section.Data()
	if err != nil {
		return "", fmt.Errorf("read ELF build ID: %w", err)
	}
	for offset := 0; offset+12 <= len(data); {
		namesz := int(file.ByteOrder.Uint32(data[offset : offset+4]))
		descsz := int(file.ByteOrder.Uint32(data[offset+4 : offset+8]))
		typeValue := file.ByteOrder.Uint32(data[offset+8 : offset+12])
		offset += 12
		nameEnd := offset + namesz
		if namesz < 0 || nameEnd > len(data) {
			break
		}
		name := bytes.TrimRight(data[offset:nameEnd], "\x00")
		offset = align4(nameEnd)
		descEnd := offset + descsz
		if descsz < 0 || descEnd > len(data) {
			break
		}
		description := data[offset:descEnd]
		offset = align4(descEnd)
		if typeValue == 3 && bytes.Equal(name, []byte("GNU")) && len(description) >= 4 {
			return hex.EncodeToString(description), nil
		}
	}
	return "", errors.New("ELF lacks a valid GNU build ID note")
}

func align4(value int) int { return (value + 3) &^ 3 }
