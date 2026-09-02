# Contributing

Contributions should preserve the portable `NeuralRenderBackend` contract and
keep exact reference paths separate from measured approximate fast paths.

Before opening a pull request:

```sh
scripts/verify.sh
```

Changes to model math need a hand-derived or independent reference test. Changes
claiming a performance improvement need paired release measurements with the
same model, input, machine, and output comparison.

Do not submit proprietary binaries, model weights, extracted shaders,
disassembly dumps, private captures, credentials, personal data, or generated
`.mlpackage`/`.nrkmodel` artifacts. Use synthetic fixtures and describe how a
maintainer can reproduce an external comparison locally.

Security-sensitive reports should follow `SECURITY.md`. Questions about whether
third-party material may be published belong in a private legal review, not a
public issue or pull request.
