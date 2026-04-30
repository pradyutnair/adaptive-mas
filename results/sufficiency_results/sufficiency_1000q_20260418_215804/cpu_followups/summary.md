# CPU Follow-ups

## 7. Calibration
- `musique`: raw ECE `0.2261` -> Platt `0.0435` -> isotonic `0.0432`; raw Brier `0.2365` -> isotonic `0.1465`
- `hotpotqa`: raw ECE `0.2017` -> Platt `0.0513` -> isotonic `0.0612`; raw Brier `0.2397` -> isotonic `0.2085`
- `2wikimultihop`: raw ECE `0.1198` -> Platt `0.0477` -> isotonic `0.0328`; raw Brier `0.1569` -> isotonic `0.1399`

## 9. Oracle Probe Upper Bound
- `musique`: oracle-answerable probes `240`; controller recall of oracle probe `0.7208`; contain `0.3660` -> `0.3760`; mean tokens `50004.3` -> `47961.3` (approx)
- `hotpotqa`: oracle-answerable probes `637`; controller recall of oracle probe `0.8885`; contain `0.6730` -> `0.6770`; mean tokens `19113.2` -> `17954.5` (approx)
- `2wikimultihop`: oracle-answerable probes `509`; controller recall of oracle probe `0.8644`; contain `0.6950` -> `0.6980`; mean tokens `32578.1` -> `31187.9` (approx)

## 11. Tau Transfer
- source sweep: MuSiQue-200 selects `tau=0.50` with contain `0.480` and mean tokens `44650.7`
- `musique`: transfer `0.50` gives contain `0.3660`, F1 `0.4114`, EM `0.2870`, mean tokens `48608.8` (approx) vs default `0.70` contain `0.3660`, tokens `50004.3`
- `hotpotqa`: transfer `0.50` gives contain `0.6740`, F1 `0.6886`, EM `0.5290`, mean tokens `18660.8` (approx) vs default `0.70` contain `0.6730`, tokens `19113.2`
- `2wikimultihop`: transfer `0.50` gives contain `0.6940`, F1 `0.6529`, EM `0.5410`, mean tokens `32172.4` (approx) vs default `0.70` contain `0.6950`, tokens `32578.1`
