# Código Antiguo / Legacy Code

Esta carpeta contiene implementaciones anteriores, notebooks de hitos del curso y código experimental que ya no se usa en la versión actual del proyecto.

---

## Contenido

### 📓 Notebooks de Hitos del Curso

| Archivo | Descripción | Fecha |
|---------|-------------|-------|
| `HitoInicial.ipynb` | Hito inicial del curso EL7024-1 | 6 Nov 2025 |
| `HitoInterme.ipynb` | Hito intermedio del curso | 6 Nov 2025 |
| `Proyecto.ipynb` | Notebook original del proyecto | 6 Nov 2025 |

**Uso:** Estos notebooks contienen las entregas de los hitos del curso. Se mantienen como referencia histórica pero ya no se usan activamente.

---

### 🐍 Scripts Python Legacy

| Archivo | Descripción | Fecha |
|---------|-------------|-------|
| `Proyecto_llm_transformado a python.py` | Conversión del notebook original a Python | 9 Nov 2025 |
| `test.py` | Scripts de testing genérico | 9 Nov 2025 |

**Uso:** Scripts de conversión y testing que fueron útiles durante desarrollo pero ya no son necesarios.

---

### 📦 Notebook Completo Antiguo

| Archivo | Descripción | Fecha |
|---------|-------------|-------|
| `proyecto_llm_uncertainty_completo.ipynb` | Análisis completo en notebook (versión antigua) | 9 Nov 2025 |

**Uso:** Versión monolítica del análisis antes de modularizar en el paquete `llm_uncertainty_analysis/`.

---

### 📁 Carpeta `src/` (Legacy Code)

| Archivo | Descripción |
|---------|-------------|
| `src/metrics.py` | Implementación original de métricas (entropy, surprisal) |
| `src/infer.py` | Código de inferencia original |
| `src/__init__.py` | Inicialización del módulo |

**Uso:** Código original del proyecto antes de la refactorización completa. Las funcionalidades fueron migradas al paquete modular `llm_uncertainty_analysis/`.

**⚠️ DEPRECADO:** Este código fue reemplazado por:
- `llm_uncertainty_analysis/metrics/` (entropy.py, surprisal.py, perplexity.py)
- `llm_uncertainty_analysis/analysis/uncertainty_analyzer.py`

---

## Código Actual del Proyecto

El código actual y activo se encuentra en:

### Scripts Principales (raíz del repo):
```
Proyecto/
├── run_multi_model_icl_analysis.py    ← Script principal del experimento
├── explore_datasets.py                 ← Exploración de LAMA, SNLI, Gutenberg
└── load_real_datasets.py              ← Carga de datasets reales
```

### Paquete Modular:
```
Proyecto/
└── llm_uncertainty_analysis/          ← Paquete Python completo
    ├── config/                        (configuración)
    ├── data/                          (loaders: LAMA, SNLI, Gutenberg)
    ├── metrics/                       (entropy, surprisal, perplexity)
    ├── analysis/                      (uncertainty_analyzer)
    ├── statistics/                    (ANOVA, effect size, MI)
    ├── icl/                           (prompt generation, entropy measurement)
    ├── visualization/                 (plots, advanced_plots)
    ├── experiments/                   (multi_model_icl_experiment)
    ├── models/                        (data models)
    └── utils/                         (helpers, reproducibility)
```

---

## Migración Completada

El código de esta carpeta (`old_code/`) fue migrado al paquete modular `llm_uncertainty_analysis/` entre Noviembre 6-9, 2025.

### Principales cambios:
1. **Modularización:** Código monolítico → paquete estructurado
2. **Datasets reales:** Hardcoded prompts → LAMA TREx, Stanford SNLI, Project Gutenberg
3. **Escalabilidad:** Experimento single-model → multi-model (3 modelos GPT-2)
4. **Testing estadístico:** Tests básicos → ANOVA, Spearman, Kendall's W, post-hoc
5. **Documentación:** Notebooks → package con README, QUICKSTART, STRUCTURE guides

---

## ¿Debo Usar Este Código?

**NO.** Este código se mantiene solo como referencia histórica.

**Para nuevos desarrollos, usa:**
- Paquete `llm_uncertainty_analysis/`
- Script `run_multi_model_icl_analysis.py`

**Para ver resultados actuales:**
- Consulta `ESTRUCTURA_REPOSITORIO.md` en la raíz

---

## Archivado

Esta carpeta puede ser excluida del control de versiones agregando `old_code/` al `.gitignore` si se desea mantener el repositorio limpio.

**Comando sugerido:**
```bash
echo "old_code/" >> .gitignore
```

Alternativamente, puede mantenerse en el repo como referencia histórica de la evolución del proyecto.
