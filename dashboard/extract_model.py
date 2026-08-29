import re, base64, os

with open('model.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

os.makedirs('assets', exist_ok=True)

pattern = re.compile(r'^\s*"([^"]+)":\s*"([A-Za-z0-9+/=]+)"')

count = 0
for line in lines:
    m = pattern.match(line)
    if m and len(m.group(2)) > 10000:
        name, b64data = m.group(1), m.group(2)
        filename = re.sub(r'\W+', '_', name.strip()).strip('_').lower() + '.glb'
        data = base64.b64decode(b64data)
        out_path = os.path.join('assets', filename)
        with open(out_path, 'wb') as out:
            out.write(data)
        print(f'Wrote assets/{filename} ({len(data)/1e6:.1f} MB) from "{name}"')
        count += 1

print(f'Done. Extracted {count} models.')