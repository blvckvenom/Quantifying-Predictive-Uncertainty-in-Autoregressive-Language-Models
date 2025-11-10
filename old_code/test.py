"""
Test script to verify imports after corrections
"""

print("Testing corrected imports...")
print("=" * 60)

# Test 1: Statistics exports (CRÍTICO)
print("\n1. Testing statistics module exports...")
try:
    from llm_uncertainty_analysis.statistics import (
        run_anova,
        calculate_eta_squared,
        run_tukey_hsd,
        run_bonferroni_correction,
        print_anova_summary,
        calculate_cohens_d,
        interpret_cohens_d,
        calculate_mutual_information,
        interpret_mutual_information
    )
    print("   ✅ All statistics functions imported successfully")
    print(f"   ✅ run_anova: {run_anova}")
    print(f"   ✅ run_tukey_hsd: {run_tukey_hsd}")
    print(f"   ✅ run_bonferroni_correction: {run_bonferroni_correction}")
except ImportError as e:
    print(f"   ❌ ERROR: {e}")
    exit(1)

# Test 2: ICL entropy reduction (RECOMENDADO)
print("\n2. Testing ICL module exports...")
try:
    from llm_uncertainty_analysis.icl import (
        generate_icl_prompt,
        measure_icl_entropy,
        measure_entropy_reduction
    )
    print("   ✅ All ICL functions imported successfully")
    print(f"   ✅ generate_icl_prompt: {generate_icl_prompt}")
    print(f"   ✅ measure_icl_entropy: {measure_icl_entropy}")
    print(f"   ✅ measure_entropy_reduction: {measure_entropy_reduction}")
except ImportError as e:
    print(f"   ❌ ERROR: {e}")
    exit(1)

# Test 3: Other critical modules
print("\n3. Testing other critical modules...")
try:
    from llm_uncertainty_analysis.config import setup_reproducibility
    from llm_uncertainty_analysis.models import DEFAULT_MODEL, MODELS
    from llm_uncertainty_analysis.metrics import calculate_entropy
    from llm_uncertainty_analysis.analysis import UncertaintyAnalyzer
    
    print("   ✅ config module: OK")
    print("   ✅ models module: OK")
    print("   ✅ metrics module: OK")
    print("   ✅ analysis module: OK")
    
    print(f"\n   📊 Default model: {DEFAULT_MODEL.name}")
    print(f"   📊 Available models: {len(MODELS)}")
    
except ImportError as e:
    print(f"   ❌ ERROR: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)