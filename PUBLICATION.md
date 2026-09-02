# Publication gate

NeuralRenderKit's generic runtime, synthetic fixtures, independently written
model graphs, and documentation are intended for an Apache-2.0 source release.
No release may contain NVIDIA binaries, extracted weights, CUDA images,
disassembly dumps, private oracle captures, or generated model packages.

This file is an engineering checklist, not legal advice. Publishing the
compatibility-research portions requires review by counsel familiar with the
publisher's jurisdiction and the agreement under which the examined software
was obtained.

## Legal review required

NVIDIA's published SDK terms restrict reverse engineering, decompilation, and
disassembly and limit redistribution to identified distributable components in
object-code applications. Article 6 of EU Directive 2009/24/EC describes a
narrow interoperability exception with conditions and restrictions, including
limits on how obtained information may be used. Counsel must determine whether
that exception, another legal basis, or no exception applies to this project and
its intended publication territories.

Primary references:

- [NVIDIA DLSS SDK license agreement](https://developer.nvidia.com/downloads/dlss/license_agreement)
- [NVIDIA Streamline repository and distribution notes](https://github.com/NVIDIA-RTX/Streamline)
- [Directive 2009/24/EC on the legal protection of computer programs](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32009L0024)

Counsel must explicitly decide whether the following may be published:

- `Tools/extract_dlssnr_weights.py` and `Tools/unpack_dlssnr_weights.py`;
- recovered constants, parameter maps, and disassembly-derived descriptions;
- compatibility naming that refers to NVIDIA or DLSS;
- converters and model topology that consume independently supplied logical
  weights.

## Engineering release checklist

- [ ] Record counsel's scope and decision without placing privileged advice in
  the public repository.
- [ ] Apply that decision to every compatibility-research file listed above.
- [ ] Run `scripts/audit-public-tree.sh .` and `scripts/verify.sh` from a clean
  checkout.
- [ ] Run the optional Python 3.12 PyTorch/Core ML tests used by CI.
- [ ] Confirm `git ls-files` contains no binary model artifact or safetensors
  outside the allowlisted synthetic fixture directory.
- [ ] Confirm `README.md`, `NOTICE`, `SECURITY.md`, and `CONTRIBUTING.md` describe
  the same package and licensing boundaries.
- [ ] Create the public repository without uploading external weights or local
  benchmark traces.
- [ ] Verify the actual remote `main` ref, then wait for CI before creating a
  release tag.

The current local tree satisfies the automated engineering checks. GitHub
publication remains gated by counsel review and valid repository credentials.
