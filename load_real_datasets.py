"""
Script para cargar los 3 datasets específicos del proyecto:
1. facebook/lama (Factual)
2. Gutenberg Poetry Corpus (Creative)
3. SNLI (Logical)

Ejecuta: python load_real_datasets.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import gzip
import json
import urllib.request
from pathlib import Path
from datasets import load_dataset
import pandas as pd

print("="*70)
print("CARGANDO DATASETS REALES PARA EL PROYECTO")
print("="*70)

# Crear carpeta para datos
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ==============================================================================
# 1. LAMA (facebook/lama) - Para contextos FACTUALES
# ==============================================================================
print("\n" + "="*70)
print("1. DATASET: facebook/lama (FACTUAL KNOWLEDGE)")
print("="*70)

def load_lama_dataset(n_samples=100):
    """Carga LAMA-TREx desde facebook/lama"""
    print("\n⏳ Cargando facebook/lama (TREx)...")

    try:
        # Intentar cargar directamente
        lama = load_dataset("facebook/lama", "trex", split="train", trust_remote_code=True)
        print(f"✅ Cargado exitosamente: {len(lama)} ejemplos")

        # Explorar estructura
        print(f"\n📋 Columnas: {lama.column_names}")
        print(f"\n📄 Ejemplo 1:")
        sample = lama[0]
        for key, value in sample.items():
            print(f"    {key}: {value}")

        # Convertir a formato del proyecto
        samples = []
        for i in range(min(n_samples, len(lama))):
            item = lama[i]
            # Crear prompt sin [MASK]
            if 'masked_sentence' in item:
                prompt = item['masked_sentence'].replace('[MASK]', '').strip()
                prompt = ' '.join(prompt.split())  # Limpiar espacios dobles
            else:
                prompt = f"{item.get('sub_label', '')} {item.get('predicate_id', '')}"

            samples.append({
                'prompt': prompt,
                'answer': item.get('obj_label', ''),
                'category': 'factual',
                'source': 'lama-trex',
                'metadata': {
                    'predicate': item.get('predicate_id', ''),
                    'subject': item.get('sub_label', '')
                }
            })

        print(f"\n✅ Convertidos {len(samples)} ejemplos al formato del proyecto")
        print(f"\n🔧 Ejemplo convertido:")
        print(f"    prompt: '{samples[0]['prompt'][:80]}...'")
        print(f"    answer: '{samples[0]['answer']}'")
        print(f"    category: '{samples[0]['category']}'")

        return samples

    except Exception as e:
        print(f"❌ Error al cargar facebook/lama: {e}")
        print(f"\n💡 Alternativa: Intentando método legacy...")

        # Fallback: Crear ejemplos sintéticos mejorados basados en LAMA
        print("⚠️ Usando ejemplos sintéticos basados en estructura LAMA")
        lama_templates = [
            ("The capital of France is", "France", "Paris"),
            ("The capital of Germany is", "Germany", "Berlin"),
            ("The capital of Italy is", "Italy", "Rome"),
            ("The capital of Spain is", "Spain", "Madrid"),
            ("The capital of Japan is", "Japan", "Tokyo"),
            ("The official language of France is", "France", "French"),
            ("The official language of Spain is", "Spain", "Spanish"),
            ("The currency of United States is", "United States", "Dollar"),
            ("The currency of United Kingdom is", "United Kingdom", "Pound"),
            ("The currency of Japan is", "Japan", "Yen"),
            ("Water freezes at", "water", "0 degrees Celsius"),
            ("The chemical symbol for gold is", "gold", "Au"),
            ("The largest planet in our solar system is", "solar system", "Jupiter"),
            ("The smallest planet in our solar system is", "solar system", "Mercury"),
            ("The Sun is a", "Sun", "star"),
        ]

        samples = []
        for i, (template, subject, answer) in enumerate(lama_templates[:n_samples]):
            samples.append({
                'prompt': template,
                'answer': answer,
                'category': 'factual',
                'source': 'lama-synthetic',
                'metadata': {'subject': subject}
            })

        print(f"✅ Creados {len(samples)} ejemplos sintéticos tipo LAMA")
        return samples

# ==============================================================================
# 2. GUTENBERG POETRY CORPUS - Para contextos CREATIVOS
# ==============================================================================
print("\n\n" + "="*70)
print("2. DATASET: Gutenberg Poetry Corpus (CREATIVE TEXT)")
print("="*70)

def load_gutenberg_poetry(n_samples=100):
    """Descarga y carga Gutenberg Poetry Corpus"""
    print("\n⏳ Descargando Gutenberg Poetry Corpus...")

    corpus_file = DATA_DIR / "gutenberg-poetry-v001.ndjson.gz"
    corpus_url = "http://static.decontextualize.com/gutenberg-poetry-v001.ndjson.gz"

    try:
        # Descargar si no existe
        if not corpus_file.exists():
            print(f"📥 Descargando desde {corpus_url}")
            print(f"   (Esto puede tardar ~1-2 minutos, ~30MB)...")
            urllib.request.urlretrieve(corpus_url, corpus_file)
            print(f"✅ Descargado en: {corpus_file}")
        else:
            print(f"✅ Ya existe en cache: {corpus_file}")

        # Leer y parsear NDJSON
        print(f"\n📖 Leyendo líneas de poesía...")
        samples = []

        with gzip.open(corpus_file, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= n_samples:
                    break

                try:
                    entry = json.loads(line)
                    poem_line = entry.get('s', '').strip()

                    # Filtrar líneas muy cortas o vacías
                    if len(poem_line.split()) >= 5:  # Al menos 5 palabras
                        # Tomar primeras 15 palabras como prompt
                        words = poem_line.split()[:15]
                        prompt = ' '.join(words)

                        samples.append({
                            'prompt': prompt,
                            'answer': None,  # Generación abierta
                            'category': 'creative',
                            'source': 'gutenberg-poetry',
                            'metadata': {
                                'gutenberg_id': entry.get('gid', ''),
                                'full_line': poem_line
                            }
                        })
                except json.JSONDecodeError:
                    continue

        print(f"✅ Cargadas {len(samples)} líneas de poesía")

        if len(samples) > 0:
            print(f"\n📄 Ejemplo 1:")
            print(f"    prompt: '{samples[0]['prompt']}'")
            print(f"    category: '{samples[0]['category']}'")
            print(f"    Gutenberg ID: {samples[0]['metadata']['gutenberg_id']}")

        return samples

    except Exception as e:
        print(f"❌ Error al cargar Gutenberg Poetry: {e}")
        print(f"\n💡 Alternativa: Usando prompts creativos sintéticos...")

        creative_prompts = [
            "In the depths of the enchanted forest where shadows dance",
            "The old wizard looked at the crystal ball and saw",
            "Under the pale moonlight the ancient castle stood",
            "She opened the mysterious letter and her heart began",
            "The storm was approaching when suddenly the lighthouse keeper",
            "Through the mist of time echoes of forgotten voices",
            "In a world where dreams become reality one must",
            "The last dragon watched from the mountain peak as",
            "Between the stars and the sea lies a realm",
            "When the clock struck midnight the enchantment began to",
        ]

        samples = []
        for prompt in creative_prompts[:n_samples]:
            samples.append({
                'prompt': prompt,
                'answer': None,
                'category': 'creative',
                'source': 'creative-synthetic',
                'metadata': {}
            })

        print(f"✅ Creados {len(samples)} prompts creativos sintéticos")
        return samples

# ==============================================================================
# 3. SNLI - Para contextos LÓGICOS
# ==============================================================================
print("\n\n" + "="*70)
print("3. DATASET: SNLI - Stanford Natural Language Inference (LOGICAL)")
print("="*70)

def load_snli_dataset(n_samples=100):
    """Carga SNLI desde Hugging Face"""
    print("\n⏳ Cargando SNLI...")

    try:
        snli = load_dataset("snli", split="train")
        print(f"✅ Cargado exitosamente: {len(snli)} ejemplos")
        print(f"\n📋 Columnas: {snli.column_names}")

        # Mostrar ejemplo
        print(f"\n📄 Ejemplo 1:")
        sample = snli[0]
        for key, value in sample.items():
            print(f"    {key}: {value}")

        # Convertir a formato del proyecto
        samples = []
        label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

        for i in range(min(n_samples * 2, len(snli))):  # Pedir más porque filtraremos
            item = snli[i]

            # Filtrar ejemplos con label válido (no -1)
            if item['label'] != -1:
                premise = item['premise']
                hypothesis = item['hypothesis']
                label = label_map.get(item['label'], 'unknown')

                # Crear prompt de inferencia lógica
                prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nRelation:"

                samples.append({
                    'prompt': prompt,
                    'answer': label,
                    'category': 'logical',
                    'source': 'snli',
                    'metadata': {
                        'premise': premise,
                        'hypothesis': hypothesis
                    }
                })

                if len(samples) >= n_samples:
                    break

        print(f"\n✅ Convertidos {len(samples)} ejemplos al formato del proyecto")
        print(f"\n🔧 Ejemplo convertido:")
        print(f"    prompt: '{samples[0]['prompt'][:80]}...'")
        print(f"    answer: '{samples[0]['answer']}'")
        print(f"    category: '{samples[0]['category']}'")

        return samples

    except Exception as e:
        print(f"❌ Error al cargar SNLI: {e}")
        print(f"\n💡 Alternativa: Usando ejemplos de inferencia lógica sintéticos...")

        logical_examples = [
            ("If x = 5 and y = 3, then x + y =", "8"),
            ("2 + 2 * 3 =", "8"),
            ("The next number in sequence 2, 4, 6, 8 is", "10"),
            ("If all cats are animals, and Tom is a cat, then Tom is", "an animal"),
            ("5 * 6 =", "30"),
            ("The square root of 16 is", "4"),
            ("10 divided by 2 equals", "5"),
            ("If today is Monday, tomorrow is", "Tuesday"),
            ("The capital letter before B is", "A"),
            ("2 to the power of 3 equals", "8"),
        ]

        samples = []
        for prompt, answer in logical_examples[:n_samples]:
            samples.append({
                'prompt': prompt,
                'answer': answer,
                'category': 'logical',
                'source': 'logical-synthetic',
                'metadata': {}
            })

        print(f"✅ Creados {len(samples)} ejemplos lógicos sintéticos")
        return samples

# ==============================================================================
# 4. CARGAR TODOS LOS DATASETS
# ==============================================================================
print("\n\n" + "="*70)
print("CARGANDO TODOS LOS DATASETS")
print("="*70)

N_SAMPLES_PER_CATEGORY = 50  # Ajusta según necesites

print(f"\n🎯 Meta: {N_SAMPLES_PER_CATEGORY} samples por categoría")

factual_data = load_lama_dataset(N_SAMPLES_PER_CATEGORY)
creative_data = load_gutenberg_poetry(N_SAMPLES_PER_CATEGORY)
logical_data = load_snli_dataset(N_SAMPLES_PER_CATEGORY)

# ==============================================================================
# 5. CONSOLIDAR Y EXPORTAR
# ==============================================================================
print("\n\n" + "="*70)
print("CONSOLIDANDO DATASETS")
print("="*70)

all_samples = factual_data + creative_data + logical_data

print(f"\n📊 Resumen:")
print(f"   Factual:  {len(factual_data)} samples")
print(f"   Creative: {len(creative_data)} samples")
print(f"   Logical:  {len(logical_data)} samples")
print(f"   TOTAL:    {len(all_samples)} samples")

# Convertir a DataFrame
df = pd.DataFrame(all_samples)

# Guardar como JSON y CSV
output_json = DATA_DIR / "consolidated_datasets.json"
output_csv = DATA_DIR / "consolidated_datasets.csv"

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(all_samples, f, ensure_ascii=False, indent=2)

df[['prompt', 'answer', 'category', 'source']].to_csv(output_csv, index=False, encoding='utf-8')

print(f"\n✅ Exportado a:")
print(f"   📄 {output_json}")
print(f"   📄 {output_csv}")

# Estadísticas por categoría
print(f"\n📈 Estadísticas por categoría:")
print(df.groupby('category').size())

print(f"\n📈 Estadísticas por fuente:")
print(df.groupby('source').size())

# Mostrar ejemplos de cada categoría
print(f"\n" + "="*70)
print("EJEMPLOS FINALES")
print("="*70)

for category in ['factual', 'logical', 'creative']:
    samples_cat = [s for s in all_samples if s['category'] == category]
    if samples_cat:
        print(f"\n🔹 {category.upper()} (ejemplo):")
        ex = samples_cat[0]
        print(f"   Prompt: {ex['prompt'][:100]}...")
        print(f"   Answer: {ex['answer']}")
        print(f"   Source: {ex['source']}")

print("\n" + "="*70)
print("✅ CARGA COMPLETADA EXITOSAMENTE")
print("="*70)
print(f"""
📝 Próximos pasos:
   1. Revisa los archivos en la carpeta 'data/'
   2. Copia la función de carga a tu notebook
   3. Reemplaza la clase DatasetManager con estos datos reales
   4. Re-ejecuta tu análisis con los datos reales
""")
