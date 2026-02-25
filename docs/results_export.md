## Results export (CSV + XLSX)

This framework can export structured per-step metrics **without parsing stdout**.

### Enable in scene JSON

Add this block:

```json
{
  "export": {
    "results": {
      "enable": true,
      "out_dir": "out",
      "base_name": "run",
      "formats": ["csv", "xlsx"],
      "every": 0
    }
  }
}
```

- **`enable`**: master switch
- **`out_dir`**: directory created if missing
- **`base_name`**: file prefix
- **`formats`**: any of `["csv","xlsx"]`
- **`every`**: if `>0`, overwrite export files every N steps (latest snapshot); always exports at end

### Files created

- **CSV**
  - `<out_dir>/<base_name>_steps.csv`
  - `<out_dir>/<base_name>_vxprof.csv`
- **Excel**
  - `<out_dir>/<base_name>.xlsx`
    - sheet `steps`
    - sheet `vx_profile`
    - sheet `meta` (timestamp, git commit hash if available, scene path, solver config, etc.)


