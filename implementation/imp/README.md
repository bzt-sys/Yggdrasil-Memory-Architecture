# SubstrateAPI compatibility drop-in

Copy `substrate_api.py` into:

```text
D:\Yggdrasil\imp_split\imp\substrate_api.py
```

Then run:

```powershell
cd D:\Yggdrasil\imp_split
python -m compileall imp
python imp/tools/run_humaneval_slice.py --limit 1 --actor none --retry-until-correct --max-attempts 1 --print-context --use-environment-bridge
```
