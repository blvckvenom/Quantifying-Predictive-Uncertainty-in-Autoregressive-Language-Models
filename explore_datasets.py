"""
Script para explorar los datasets antes de usarlos en el notebook
Ejecuta: python explore_datasets.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from datasets import load_dataset
import pandas as pd

print("="*60)
print("EXPLORANDO DATASETS PARA EL PROYECTO")
print("="*60)

# ==============================================================================
# 1. LAMA - Para contextos FACTUALES
# ==============================================================================
print("\n" + "="*60)
print("1. DATASET LAMA (Factual Knowledge)")
print("="*60)

print("\n⏳ Cargando LAMA-TREx...")
try:
    # TREx es un subset de LAMA con hechos de Wikidata
    lama = load_dataset("lama", "trex", split="train")

    print(f"✅ Cargado exitosamente: {len(lama)} ejemplos")
    print(f"\n📋 Columnas disponibles: {lama.column_names}")

    # Mostrar primeros 5 ejemplos
    print("\n📄 Primeros 5 ejemplos:")
    for i in range(min(5, len(lama))):
        item = lama[i]
        print(f"\n  Ejemplo {i+1}:")
        print(f"    UUID: {item.get('uuid', 'N/A')}")
        print(f"    Relación: {item.get('predicate_id', 'N/A')}")
        print(f"    Sujeto: {item.get('sub_label', 'N/A')}")
        print(f"    Objeto: {item.get('obj_label', 'N/A')}")
        print(f"    Template: {item.get('template', 'N/A')}")
        if 'masked_sentence' in item:
            print(f"    Frase con [MASK]: {item['masked_sentence']}")

    # Convertir a formato usable para tu proyecto
    print("\n🔧 Formato sugerido para tu proyecto:")
    sample = lama[0]
    if 'masked_sentence' in sample:
        # Remover [MASK] para generar el prompt
        prompt = sample['masked_sentence'].replace('[MASK]', '').replace('  ', ' ').strip()
        answer = sample.get('obj_label', '')
        print(f"    prompt: '{prompt}'")
        print(f"    answer: '{answer}'")
        print(f"    category: 'factual'")

    # Estadísticas
    print(f"\n📊 Estadísticas:")
    print(f"    Total ejemplos: {len(lama)}")
    if 'predicate_id' in lama.column_names:
        predicates = pd.Series([item['predicate_id'] for item in lama])
        print(f"    Relaciones únicas: {predicates.nunique()}")
        print(f"\n    Top 5 relaciones más comunes:")
        print(predicates.value_counts().head().to_string())

except Exception as e:
    print(f"❌ Error al cargar LAMA: {e}")
    print(f"    Alternativa: Usar 'lama' sin subset especificado")

# ==============================================================================
# 2. GSM8K - Para contextos LÓGICOS/MATEMÁTICOS
# ==============================================================================
print("\n\n" + "="*60)
print("2. DATASET GSM8K (Grade School Math)")
print("="*60)

print("\n⏳ Cargando GSM8K...")
try:
    gsm8k = load_dataset("gsm8k", "main", split="train")

    print(f"✅ Cargado exitosamente: {len(gsm8k)} ejemplos")
    print(f"\n📋 Columnas disponibles: {gsm8k.column_names}")

    # Mostrar primeros 3 ejemplos
    print("\n📄 Primeros 3 ejemplos:")
    for i in range(min(3, len(gsm8k))):
        item = gsm8k[i]
        print(f"\n  Ejemplo {i+1}:")
        print(f"    Pregunta: {item['question'][:100]}...")
        print(f"    Respuesta completa: {item['answer'][:150]}...")
        # Extraer solo el número final (después de ####)
        if '####' in item['answer']:
            final_answer = item['answer'].split('####')[-1].strip()
            print(f"    Respuesta numérica: {final_answer}")

    # Formato para tu proyecto
    print("\n🔧 Formato sugerido para tu proyecto:")
    sample = gsm8k[0]
    prompt = sample['question']
    answer = sample['answer'].split('####')[-1].strip() if '####' in sample['answer'] else sample['answer']
    print(f"    prompt: '{prompt}'")
    print(f"    answer: '{answer}'")
    print(f"    category: 'logical'")

    print(f"\n📊 Estadísticas:")
    print(f"    Total problemas: {len(gsm8k)}")
    # Analizar longitud de preguntas
    lengths = [len(item['question'].split()) for item in gsm8k]
    print(f"    Longitud promedio pregunta: {sum(lengths)/len(lengths):.1f} palabras")

except Exception as e:
    print(f"❌ Error al cargar GSM8K: {e}")
    print(f"    Tip: Este dataset es público en Hugging Face")

# ==============================================================================
# 3. WRITINGPROMPTS - Para contextos CREATIVOS
# ==============================================================================
print("\n\n" + "="*60)
print("3. DATASET WRITINGPROMPTS (Creative Text)")
print("="*60)

print("\n⏳ Cargando WritingPrompts...")
try:
    # Cargar solo una muestra pequeña para exploración (es muy grande)
    writing = load_dataset("writingprompts", split="train[:1000]")

    print(f"✅ Cargado exitosamente: {len(writing)} ejemplos (muestra)")
    print(f"\n📋 Columnas disponibles: {writing.column_names}")

    # Mostrar primeros 3 ejemplos
    print("\n📄 Primeros 3 ejemplos:")
    for i in range(min(3, len(writing))):
        item = writing[i]
        prompt_text = item.get('prompt', '')
        # Limpiar prompt (quitar prefijos comunes de Reddit)
        if prompt_text.startswith('['):
            prompt_text = ' '.join(prompt_text.split()[1:])  # Quitar [WP], [EU], etc.

        print(f"\n  Ejemplo {i+1}:")
        print(f"    Prompt: {prompt_text[:150]}...")
        if 'story' in item:
            print(f"    Historia: {item['story'][:100]}...")

    # Formato para tu proyecto
    print("\n🔧 Formato sugerido para tu proyecto:")
    sample = writing[0]
    prompt = sample.get('prompt', '').split()
    if prompt[0].startswith('['):
        prompt = ' '.join(prompt[1:])  # Quitar tag
    else:
        prompt = ' '.join(prompt)
    # Truncar a longitud razonable
    prompt = ' '.join(prompt.split()[:30])

    print(f"    prompt: '{prompt}'")
    print(f"    answer: None  # (generación abierta)")
    print(f"    category: 'creative'")

    print(f"\n📊 Estadísticas:")
    print(f"    Total prompts en dataset completo: ~300,000")
    print(f"    Muestra cargada: {len(writing)}")

except Exception as e:
    print(f"❌ Error al cargar WritingPrompts: {e}")
    print(f"    Alternativa: Usar 'story' field para prompts")

# ==============================================================================
# 4. RESUMEN Y RECOMENDACIONES
# ==============================================================================
print("\n\n" + "="*60)
print("RESUMEN Y RECOMENDACIONES")
print("="*60)

print("""
✅ Datasets identificados:
   1. LAMA (lama/trex) → Factual knowledge
   2. GSM8K (gsm8k/main) → Logical reasoning
   3. WritingPrompts → Creative generation

📝 Estructura común que necesitas:
   {
       "prompt": str,      # El texto de entrada
       "answer": str|None, # Respuesta esperada (None para creative)
       "category": str     # "factual", "logical", "creative"
   }

💡 Recomendaciones de muestreo:
   - LAMA: 50-100 ejemplos de diferentes relaciones
   - GSM8K: 50-100 problemas variados
   - WritingPrompts: 50-100 prompts cortos (<30 palabras)

⚡ Para empezar rápido:
   1. Ejecuta este script para verificar que los datasets cargan
   2. Copia el código de carga a tu notebook
   3. Adapta los campos a tu clase DatasetManager

🔗 Documentación:
   - Hugging Face Datasets: https://huggingface.co/docs/datasets
   - LAMA: https://huggingface.co/datasets/lama
   - GSM8K: https://huggingface.co/datasets/gsm8k
   - WritingPrompts: https://huggingface.co/datasets/writingprompts
""")

print("\n" + "="*60)
print("✅ EXPLORACIÓN COMPLETADA")
print("="*60)
