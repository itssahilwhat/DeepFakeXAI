#!/usr/bin/env bash
# Create venv in project root
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv $VENV_DIR
fi
source $VENV_DIR/bin/activate

echo "Installing requirements..."
$PIP install --upgrade pip
if [ -f "requirements.txt" ]; then
    $PIP install -r requirements.txt
else
    echo "❌ requirements.txt not found."
    deactivate
    exit 1
fi

echo "✅ Setup complete."
echo
echo "To activate the environment later, run: source $VENV_DIR/bin/activate"
echo "Run Commands (from project root):"
echo "1. Train: python src/train.py"
echo "2. Distill: python scripts/distillation.py --teacher path/to/teacher.pth --student_out checkpoints/student.pth"
echo "3. Robustness test: python scripts/test_robustness.py --dataset celebahq --subset test"
echo "4. Fairness eval: python scripts/evaluate_fairness.py -i path/to/results.json -o outputs/fairness_breakdown.json"
echo "5. Benchmark: python scripts/benchmark_speed.py --dataset celebahq"
echo "6. API (FastAPI): uvicorn api.fastapi_app:app --reload"
echo "7. Flask UI: python api/src/main.py"
echo "8. End-to-end test: python src/test_system.py --mode test --folder path/to/samples"
echo "9. ONNX export: curl -X POST http://localhost:8000/export_onnx"
