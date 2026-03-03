## Results export (CSV + XLSX)

This framework can export structured per-step metrics **without parsing stdout**.

### VX profile diagnostics mode

`solver.debug_vx_profile.y_extent_mode` controls how the y-extent for profile
bins is selected:

- `config` (default): strict benchmark mode using configured `y0`/`H`
- `walls`: preferred validation mode using wall/domain extents (`y0_wall`, `H_wall`)
- `slice_auto`: robust diagnostics mode using the sampled set after x-slice
  filtering (`y0_eff=min(y)`, `H_eff=max(y)-min(y)`)

Additional key:
- `avg_window_steps`: rolling averaging window (in profile-sample steps) used
  for `[VXERR_AVG]` to reduce transient noise.

Logs include:

- `[VXPROF_WALL] ... y0_wall=... H_wall=... y_wall_range=[...]` (walls mode)
- `[VXPROF_AUTO] ... y0_eff=... H_eff=... y_world_range=[...]`
- `[VXANA] ... mode=<config|walls|slice_auto> ...`
- `[VXERR] ... empty_bins=... used_bins=k/total ...`
- `[VXERR_AVG] ... window=current/avg_window_steps ...`

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


