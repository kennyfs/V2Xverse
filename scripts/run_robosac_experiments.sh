#!/bin/bash
# Run ROBOSAC experimental conditions on morecav fine-tuned model.
# The attack script handles both no-def and ROBOSAC-def in one pass.
# Run from V2Xverse root with: conda activate v2xverse

MODEL_DIR=opencood/logs/morecav_finetune_2026_05_28

# --- C1: No attack, no defense (upper-bound baseline) ---
echo ""
echo "=========================================="
echo "C1: No attack, no defense"
echo "=========================================="
python opencood/tools/inference_multiclass_robosac.py \
    --model_dir $MODEL_DIR --no_robosac
cp ${MODEL_DIR}/AP_robosac.txt   ${MODEL_DIR}/AP_C1_no_attack_nodef.txt 2>/dev/null || true
cp ${MODEL_DIR}/AP_robosac030507.yaml ${MODEL_DIR}/AP_C1.yaml 2>/dev/null || true

# --- C2: No attack + ROBOSAC defense ---
echo ""
echo "=========================================="
echo "C2: No attack + ROBOSAC"
echo "=========================================="
python opencood/tools/inference_multiclass_robosac.py \
    --model_dir $MODEL_DIR
cp ${MODEL_DIR}/AP_robosac.txt   ${MODEL_DIR}/AP_C2_no_attack_robosac.txt 2>/dev/null || true
cp ${MODEL_DIR}/AP_robosac030507.yaml ${MODEL_DIR}/AP_C2.yaml 2>/dev/null || true

# --- C3+C4: FP attack (script runs no-def and ROBOSAC-def in one pass) ---
echo ""
echo "=========================================="
echo "C3+C4: FP attack, no-def + ROBOSAC"
echo "=========================================="
python opencood/tools/inference_multiclass_attack_robosac.py \
    --model_dir $MODEL_DIR --attack_goal fp

# --- C5+C6: FN attack (script runs no-def and ROBOSAC-def in one pass) ---
echo ""
echo "=========================================="
echo "C5+C6: FN attack, no-def + ROBOSAC"
echo "=========================================="
python opencood/tools/inference_multiclass_attack_robosac.py \
    --model_dir $MODEL_DIR --attack_goal fn

# --- Summary ---
echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="

for label in C1_no_attack_nodef C2_no_attack_robosac; do
    f=${MODEL_DIR}/AP_${label}.txt
    if [ -f "$f" ]; then
        echo "${label}: $(cat $f)"
    else
        echo "${label}: MISSING"
    fi
done

for f in ${MODEL_DIR}/AP_attack*.txt; do
    [ -f "$f" ] && echo "$(basename $f .txt): $(cat $f)"
done
