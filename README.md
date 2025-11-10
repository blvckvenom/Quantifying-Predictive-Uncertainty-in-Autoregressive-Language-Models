# LLM Uncertainty Analysis

Análisis cuantitativo de incertidumbre predictiva en modelos de lenguaje autoregresivos utilizando teoría de la información. Este proyecto en instancia intermedia se centra en cómo el **In-Context Learning (ICL)** afecta la entropía predictiva en diferentes categorías de tareas.

**Curso:** EL7024-1 Teoría de la Información (2025-2)
**Universidad:** Universidad de Chile
**Autores:** Benito Fuentes, Sebastian Vergara

---

## Descripción del Proyecto

Este repositorio contiene una implementación modular para medir y analizar la **incertidumbre predictiva** en modelos de lenguaje (GPT-2 family) usando **entropía de Shannon** como métrica principal. El proyecto evalúa cómo el aprendizaje en contexto (ICL) reduce la incertidumbre del modelo en tres categorías de tareas:

1. **Conocimiento Factual** (LAMA TREx)
2. **Razonamiento Lógico** (Stanford SNLI)
3. **Generación Creativa** (Project Gutenberg Poetry)

### Hallazgos Principales

- **Razonamiento lógico (SNLI)** muestra la mayor reducción de entropía: +4.75 bits (71% reducción en DistilGPT-2)
- **Conocimiento factual (LAMA)** exhibe **entropía negativa** en modelos pequeños: -3.23 bits en DistilGPT-2 (ICL contraproducente)
- **Generación creativa (Gutenberg)** permanece neutral: ΔH ≈ 0 bits
- **Escalamiento dependiente de tarea**: SNLI favorece modelos pequeños (ρ=-1.0), LAMA favorece modelos grandes (ρ=+1.0)

---

## Estructura del Repositorio

### 📂 Scripts Principales (Raíz)

| Archivo | Descripción |
|---------|-------------|
| **`run_multi_model_icl_analysis.py`** | Script principal del experimento. Ejecuta análisis multi-modelo ICL en 3 modelos GPT-2 con 1,800 prompts de benchmarks reales. Genera resultados, figuras y análisis estadístico completo. |
| **`explore_datasets.py`** | Herramienta de exploración de datasets. Muestra estadísticas detalladas de LAMA TREx (1,000 prompts factuales), Stanford SNLI (300 pares lógicos) y Project Gutenberg (500 líneas de poesía). |
| **`load_real_datasets.py`** | Funciones auxiliares para cargar y procesar los tres datasets reales utilizados en el proyecto. Provee API unificada para acceso a datos. |

### 📦 Paquete `llm_uncertainty_analysis/`

Paquete Python modular que implementa todo el pipeline de análisis de incertidumbre:

```
llm_uncertainty_analysis/
├── config/              # Configuración global y de visualización
├── data/                # Loaders para LAMA, SNLI, Gutenberg
├── metrics/             # Entropía de Shannon, Surprisal, Perplexity
├── analysis/            # Pipeline completo de análisis de incertidumbre
├── statistics/          # ANOVA, Spearman, Kendall's W, Cohen's d
├── icl/                 # Generación de prompts ICL y medición de entropía
├── visualization/       # Generación de gráficos estadísticos
├── experiments/         # Experimento multi-modelo ICL (script principal)
├── models/              # Modelos de datos y configuraciones
└── utils/               # Utilidades y reproducibilidad
```

**Documentación del paquete:** Ver `llm_uncertainty_analysis/README.md`

### 🗄️ Carpeta `data/`

Contiene los tres datasets de benchmarks reales utilizados en el proyecto:

- **`lama_data/`**: LAMA TREx (Language Model Analysis)
  - 1,000 prompts de conocimiento factual
  - 4 relaciones Wikidata: P19 (nacimiento), P37 (idioma), P106 (ocupación), P36 (capital)

- **`gutenberg-poetry-v001.ndjson.gz`**: Project Gutenberg Poetry
  - 3.08M+ líneas de poesía de 1,191 libros clásicos
  - Muestreo estratificado: 500 líneas de 50 obras (1800-1922)

- **`consolidated_datasets.csv/json`**: Datasets consolidados
  - Incluye 300 prompts de Stanford SNLI
  - Pares premise-hypothesis balanceados (entailment/neutral/contradiction)

### 📈 Carpeta `outputs/`

Resultados de experimentos en formato JSON:

```
outputs/
└── multi_model_icl/
    ├── results.json                  # Resultados completos del experimento
    ├── statistical_analysis.json     # ANOVA, correlaciones, post-hoc tests
    └── hypothesis_validation.json    # Validación de H1 (scaling) y H2 (consistency)
```

### 📊 Carpeta `out/`

Resultados intermedios y métricas token-level en formato CSV:

- `etapa2_tokens_metrics_all_models.csv`: Métricas por token para todos los modelos
- `etapa2_agregados_por_modelo.csv`: Agregados estadísticos por modelo
- `etapa2_agregados_por_texto.csv`: Agregados por texto/prompt
- `etapa2_benchmark.json`: Benchmarks de rendimiento

### 📉 Carpeta `fig/`

Visualizaciones generadas (formato PNG):

- `icl_comprehensive_analysis.png`: Análisis comprensivo de ICL
- `entropy_by_category_*.png`: Comparaciones de entropía por categoría
- `icl_mutual_information_heatmap.png`: Heatmap de información mutua
- `etapa2_*.png`: Visualizaciones de análisis intermedios

### 🗃️ Carpeta `old_code/`

Código legacy del proyecto (notebooks de hitos, implementaciones anteriores). Ver `old_code/README.md` para detalles.

---

## Instalación

### Requisitos

- Python 3.8+
- CUDA-capable GPU (opcional, recomendado para experimentos grandes)

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone <repository-url>
cd Proyecto

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

#### Dependencias Principales

- **transformers** (4.44.2): Modelos Hugging Face (GPT-2)
- **torch** (>=2.3.0): PyTorch para inferencia
- **numpy**, **pandas**: Procesamiento de datos
- **matplotlib**, **seaborn**: Visualización
- **scipy**, **statsmodels**: Análisis estadístico
- **datasets**: Carga de SNLI desde Hugging Face

Para soporte GPU, instalar PyTorch con CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Uso

### Ejecutar Experimento Principal

```bash
# Experimento completo (1,800 prompts, ~17 segundos en GPU)
python run_multi_model_icl_analysis.py

# Experimento rápido (subset de datos para testing)
python run_multi_model_icl_analysis.py --quick
```

**Salida:**
- Resultados JSON en `outputs/multi_model_icl/`
- Figuras PNG en `fig/`
- Análisis estadístico completo (ANOVA, correlaciones)

### Explorar Datasets

```bash
# Mostrar estadísticas de LAMA, SNLI, Gutenberg
python explore_datasets.py
```

**Salida:**
```
LAMA TREx: 1,000 prompts factuales (4 relaciones Wikidata)
Stanford SNLI: 300 pares lógicos (balanceados)
Project Gutenberg: 500 líneas de poesía (50 libros)
```

### Usar el Paquete en Python

```python
from llm_uncertainty_analysis.experiments import run_multi_model_icl_experiment

# Ejecutar experimento personalizado
results = run_multi_model_icl_experiment(
    model_ids=['distilgpt2', 'gpt2', 'gpt2-medium'],
    n_examples_range=[0, 1, 2, 3, 5],  # k-shot configurations
    n_queries_per_config=10,
    device='cuda'
)

# Acceder a resultados
print(results['results_by_model']['distilgpt2']['categories']['factual'])
```

---

## Datasets

### LAMA TREx (Factual Knowledge)

**Fuente:** Petroni et al. 2019 - "Language Models as Knowledge Bases?"

- **Tamaño:** 1,000 prompts
- **Relaciones Wikidata:**
  - P19: Lugar de nacimiento (250 prompts)
  - P37: Idioma oficial (250 prompts)
  - P106: Ocupación (250 prompts)
  - P36: Capital (250 prompts)
- **Formato:** `"The official language of France is"` → `"French"`

### Stanford SNLI (Logical Reasoning)

**Fuente:** Bowman et al. 2015 - "A large annotated corpus for learning natural language inference"

- **Tamaño:** 300 pares premise-hypothesis
- **Labels balanceados:**
  - Entailment: 100 pares
  - Neutral: 100 pares
  - Contradiction: 100 pares
- **Formato:** `"Premise: ... Hypothesis: ... Relation:"` → `"entailment"`

### Project Gutenberg (Creative Generation)

**Fuente:** Project Gutenberg 2024

- **Tamaño:** 500 líneas de poesía
- **Obras:** 50 libros clásicos (1800-1922)
- **Muestreo:** Estratificado (10 líneas por libro)
- **Formato:** Predicción de siguiente línea poética

---

## Modelos Analizados

| Modelo | Parámetros | VRAM | Descripción |
|--------|------------|------|-------------|
| **DistilGPT-2** | 82M | 0.3 GB | Versión destilada de GPT-2, compresión de conocimiento |
| **GPT-2** | 124M | 0.5 GB | GPT-2 Small original, modelo baseline |
| **GPT-2 Medium** | 355M | 1.5 GB | GPT-2 Medium, mayor capacidad |

**Rango:** 4.3× en número de parámetros (82M - 355M)

---

## Métricas

### Entropía de Shannon (H)

```python
H = -Σ p(x) * log₂(p(x))
```

Mide la **incertidumbre** del modelo sobre la distribución completa de probabilidad del siguiente token.

- **Unidad:** bits
- **Interpretación:** Bits de información necesarios para codificar la distribución
- **Rango:** 0 (certeza total) a log₂(|V|) (incertidumbre máxima)

### Reducción de Entropía (ΔH)

```python
ΔH = H(0-shot) - H(k-shot)
```

Mide la **efectividad del In-Context Learning** como reducción de incertidumbre.

- **ΔH > 0:** ICL reduce incertidumbre (efectivo)
- **ΔH = 0:** ICL no tiene efecto
- **ΔH < 0:** ICL aumenta incertidumbre (contraproducente)

### Métricas Implementadas (No Usadas en Experimento ICL)

- **Surprisal:** `S = -log₂(p(y_true))` - Sorpresa ante token específico
- **Perplexity:** `PPL = 2^S` - "Número efectivo de opciones equiprobables"

---

## Experimento

### Diseño Experimental

**Configuraciones k-shot:** [0, 1, 2, 3, 5]

Para cada combinación de:
- **Modelo:** DistilGPT-2, GPT-2, GPT-2 Medium
- **Categoría:** Factual (LAMA), Logical (SNLI), Creative (Gutenberg)
- **k-shot:** 0, 1, 2, 3, 5 ejemplos

Se mide:
1. Entropía predictiva H del primer token de respuesta
2. Reducción de entropía ΔH respecto a baseline (0-shot)
3. Análisis estadístico: ANOVA, Spearman, Kendall's W

### Hipótesis Evaluadas

**H1 (Scaling):** "Modelos más grandes muestran mayor efectividad ICL"
- **Resultado:** ❌ NO SOPORTADA
- **Evidencia:** Escalamiento es dependiente de tarea
  - SNLI: ρ = -1.0 (modelos pequeños mejor)
  - LAMA: ρ = +1.0 (modelos grandes mejor)

**H2 (Consistency):** "El ranking de categorías es consistente entre modelos"
- **Resultado:** ✅ SOPORTADA
- **Evidencia:** Kendall's W = 1.000, p = 0.0498
- **Ranking:** logical > creative > factual (en todos los modelos)

---

## Resultados

### Reducción de Entropía por Categoría (5-shot)

| Modelo | Factual (LAMA) | Logical (SNLI) | Creative (Gutenberg) |
|--------|----------------|----------------|----------------------|
| **DistilGPT-2** | -3.23 bits (-46%) | **+4.75 bits (+71%)** | -0.13 bits (-2%) |
| **GPT-2** | -1.65 bits (-23%) | **+4.25 bits (+51%)** | -0.21 bits (-3%) |
| **GPT-2 Medium** | -0.54 bits (-7%) | **+1.86 bits (+23%)** | +0.20 bits (+2%) |

**Significancia:** \*\*\* p < 0.001, \*\* p < 0.01, ns = not significant

### Interpretación

1. **SNLI (Logical) es la categoría más efectiva para ICL**
   - Formato estructurado premise-hypothesis permite aprendizaje rápido
   - Incluso modelos pequeños (82M parámetros) logran 71% de reducción

2. **LAMA (Factual) muestra ΔH negativo en modelos pequeños**
   - Ejemplos multi-relación (P19/P37/P106/P36 mezclados) confunden modelos con capacidad limitada
   - Solo modelos grandes (355M+) manejan la diversidad

3. **Gutenberg (Creative) permanece de alta entropía**
   - Múltiples continuaciones poéticas igualmente válidas
   - ICL no puede reducir incertidumbre inherente

4. **Escalamiento NO es universal**
   - SNLI: Modelos pequeños superan a grandes (maleabilidad)
   - LAMA: Modelos grandes superan a pequeños (capacidad)

---

## Análisis Estadístico

### ANOVA de Dos Vías

**Efecto de Categoría:** F = 16.15, p = 0.0038 (**significativo**)
- Las categorías difieren significativamente en efectividad ICL

**Efecto de Modelo:** F = 0.01, p = 0.9894 (no significativo)
- No hay efecto global del tamaño de modelo (es dependiente de categoría)

### Correlaciones de Spearman

**Por Categoría:**
- Factual (LAMA): ρ = +1.0 (escalamiento positivo perfecto)
- Logical (SNLI): ρ = -1.0 (escalamiento inverso perfecto)
- Creative (Gutenberg): ρ = +0.5 (no significativo)

### Consistencia de Ranking (Kendall's W)

**W = 1.000, p = 0.0498**
- Acuerdo perfecto entre modelos en el ranking de categorías
- Todos los modelos coinciden: SNLI > Gutenberg > LAMA

---

## Visualizaciones

Las figuras generadas se encuentran en `fig/`:

- **ICL Comprehensive Analysis:** Comparación de entropía 0-shot vs 5-shot por categoría
- **Entropy by Category:** Distribuciones de entropía con intervalos de confianza
- **Mutual Information Heatmap:** Información mutua entre ejemplos ICL y predicción
- **Scaling Analysis:** Correlaciones Spearman entre tamaño de modelo y ΔH

---

## Documentación Adicional

- **`llm_uncertainty_analysis/README.md`**: Documentación del paquete modular
- **`llm_uncertainty_analysis/QUICKSTART.md`**: Guía rápida de uso
- **`llm_uncertainty_analysis/STRUCTURE.md`**: Estructura detallada del código
- **`llm_uncertainty_analysis/MIGRATION_GUIDE.md`**: Guía de migración de código legacy
- **`old_code/README.md`**: Documentación del código antiguo

---

## Referencias

### Datasets

- **LAMA TREx:** Petroni, F., et al. (2019). "Language models as knowledge bases?" *arXiv:1909.01066*.
- **Stanford SNLI:** Bowman, S. R., et al. (2015). "A large annotated corpus for learning natural language inference." *EMNLP 2015*.
- **Project Gutenberg:** [https://www.gutenberg.org/](https://www.gutenberg.org/)

### Métricas

- **Entropía & Surprisal:** Levy, R. (2008). "Expectation-based syntactic comprehension." *Cognition*, 106(3), 1126-1177.
- **Perplexity:** Goodkind, A., & Bicknell, K. (2018). "Predictive power of word surprisal for reading times is a linear function of language model quality." *CMCL 2018*.

---

## Licencia

Este proyecto fue desarrollado como parte del curso EL7024-1 Teoría de la Información, Universidad de Chile (2025-2).

---

## Autores

**Benito Fuentes**
- Diseño experimental, implementación de medición de entropía, análisis estadístico completo

**Sebastian Vergara**
- Pipeline multi-modelo, generación de visualizaciones, validación de hipótesis

**Curso:** EL7024-1 2025-2
**Profesor:** Jorge Silva
**Guía:** Simón Vidal
