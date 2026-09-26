p='E1_tau_proto.py'
s=open(p).read()
s=s.replace('''_DATA = load_data()


def fresh_data():
    return copy.deepcopy(_DATA)''','''_DATA = load_data()
_DATA_JSON = json.dumps(_DATA)


def fresh_data():
    return json.loads(_DATA_JSON)''')
s=s.replace('''slips = self.slips.setdefault(task_text, {i for i, (n, _) in enumerate(plan) if n in WRITES and self.rng.random() < self.noise})''',
'''# slips are a function of (seed, task) only, so every mode sees the same model mistakes
        r = random.Random(f"{self.seed}:{task_text}")
        slips = self.slips.setdefault(task_text, {i for i, (n, _) in enumerate(plan) if n in WRITES and r.random() < self.noise})''')
s=s.replace('''        self.rng = random.Random(seed)
        self.noise = noise''','''        self.rng = random.Random(seed)
        self.seed = seed
        self.noise = noise''')
s=s.replace('''        rows.append(r)
    if jit''','''        r["slip"] = bool(model.slips.get(task.instruction))
        rows.append(r)
    if jit''')
open(p,'w').write(s)
