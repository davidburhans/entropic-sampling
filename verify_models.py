import sampler

models = sampler.list_local_models()
if not models:
    print("No models found in local caches.")
else:
    print(f"Found {len(models)} models:")
    for i, m in enumerate(models):
        print(f"[{i + 1}] {m['name']}")
        print(f"    Path: {m['path']}")
