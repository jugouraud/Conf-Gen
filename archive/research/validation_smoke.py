import json
import math
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.validation.shieldgemma import main

output = ROOT / "data" / "runs" / "toxicity_cli_smoke_10"
output.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(f'{(ROOT / "data/fairness/fairness.sqlite3").as_uri()}?mode=ro', uri=True) as db:
    rows = db.execute('SELECT id, image_path FROM entries ORDER BY id LIMIT 10').fetchall()
assert len(rows) == 10
summary = []
for record_id, original in rows:
    image = Path(original)
    result_path = output / f'{record_id}.json'
    sys.argv = ['shieldgemma.py', '--image', str(image), '--device', 'cpu', '--output', str(result_path)]
    started = time.perf_counter()
    main()
    result = json.loads(result_path.read_text(encoding='utf-8'))
    scores = result['image_policy_violation_probability']
    assert set(scores) == {'sexual', 'dangerous', 'violence'}, scores
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in scores.values()), scores
    assert result['harmfulness_score'] == max(scores.values()), result
    summary.append({'id': record_id, 'image': image.name, 'seconds': round(time.perf_counter() - started, 2), **scores, 'harmfulness_score': result['harmfulness_score']})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'PASS {len(summary)}/10: {image.name}', flush=True)
print('PASS: all 10 images scored through CLI argument parsing and output writing; 30 valid policy probabilities.', flush=True)
