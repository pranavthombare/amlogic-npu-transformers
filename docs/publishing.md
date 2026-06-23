# Publishing Checklist

This repo is structured for GitHub, including CI at `.github/workflows/ci.yml`.

Before pushing publicly:

1. Choose and add a license.
2. Confirm `make test` is green.
3. Confirm `git status --ignored --short` does not show private SDK files,
   Hugging Face tokens, model weights, converted `.nb` graphs, or generated NPU
   binaries outside ignored paths.
4. Keep board-generated artifacts out of git unless a later release explicitly
   defines an artifact publishing policy.

Suggested first push:

```sh
git remote add origin git@github.com:<owner>/vim3-gemma-npu.git
git push -u origin main
```
