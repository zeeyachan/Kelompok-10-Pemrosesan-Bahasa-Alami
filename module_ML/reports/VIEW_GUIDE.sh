#!/bin/bash
# Quick commands to view visualization dan reports

echo "🎯 ARXIV PAPER MATERIALS - QUICK VIEW GUIDE"
echo "==========================================="
echo ""
echo "📁 Folder Structure:"
echo "  module_ML/reports/"
echo "    ├── arxiv_report.json           (Paper metadata)"
echo "    ├── metrics_comparison.png      (Figure 1: Performance chart)"
echo "    ├── confusion_matrices.png      (Figure 2: Error analysis)"
echo "    ├── model_summary.png           (Figure 3: Specs)"
echo "    ├── baseline_logreg_metrics.json"
echo "    ├── baseline_svm_metrics.json"
echo "    ├── baseline_nb_metrics.json"
echo "    └── transformer_metrics.json"
echo ""
echo "📊 MODEL STRUCTURE:"
echo "  module_ML/models/"
echo "    ├── baseline/                   (TF-IDF + LogReg, SVM, Naive Bayes)"
echo "    │   ├── tfidf_logreg.joblib"
echo "    │   ├── tfidf_svm.joblib"
echo "    │   └── tfidf_nb.joblib"
echo "    └── transformer/               (IndoBERT Transformer) ← NEW!"
echo "        ├── final_model/"
echo "        └── checkpoints/"
echo ""
echo "═══════════════════════════════════════════════════════"
echo ""
echo "1️⃣  VIEW PNG IMAGES (Visualizations)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Option A: Open in VS Code (Recommended)"
echo "$ cd /workspaces/pba2026-kelompok10"
echo "$ code module_ML/reports/metrics_comparison.png"
echo ""
echo "Option B: Terminal Viewer"
echo "$ eog module_ML/reports/metrics_comparison.png"
echo "$ display module_ML/reports/confusion_matrices.png"
echo ""
echo "Option C: Python Script"
cat << 'PYEOF'
from PIL import Image
import matplotlib.pyplot as plt

# Load and display image
img = Image.open('module_ML/reports/metrics_comparison.png')
plt.figure(figsize=(12, 6))
plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.show()
PYEOF
echo ""
echo ""
echo "2️⃣  VIEW JSON FILES (Data & Metrics)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Option A: Pretty Print dengan jq (install: apt-get install jq)"
echo "$ cat module_ML/reports/arxiv_report.json | jq ."
echo "$ cat module_ML/reports/arxiv_report.json | jq '.models'"
echo ""
echo "Option B: Python Script"
cat << 'PYEOF'
import json

with open('module_ML/reports/arxiv_report.json', 'r') as f:
    report = json.load(f)

# Pretty print
print(json.dumps(report, indent=2, ensure_ascii=False))

# Access specific data
print("\n=== Baseline Model ===")
baseline = report['models']['baseline']
print(f"Accuracy:  {baseline['metrics']['accuracy']:.4f}")
print(f"Macro F1:  {baseline['metrics']['macro_f1']:.4f}")

print("\n=== Deep Learning Model ===")
dl = report['models']['deep_learning']
print(f"Accuracy:  {dl['metrics']['accuracy']:.4f}")
print(f"Macro F1:  {dl['metrics']['macro_f1']:.4f}")
PYEOF
echo ""
echo "Option C: Command Line (Linux/WSL)"
echo "$ less module_ML/reports/arxiv_report.json"
echo ""
echo ""
echo "3️⃣  VIEW FILES IN VS CODE EXPLORER"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Open VS Code Explorer (Ctrl+Shift+E)"
echo "2. Navigate to: module_ML/reports/"
echo "3. Click on:"
echo "   • metrics_comparison.png → Preview pane shows image"
echo "   • arxiv_report.json → VS Code formats JSON nicely"
echo "   • model_summary.png → Click to view full size"
echo ""
echo "💡 Tip: Right-click image → 'Open with External App'"
echo ""
echo ""
echo "4️⃣  CREATE WEB VIEW (For sharing)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Start simple HTTP server:"
echo "$ cd module_ML/reports"
echo "$ python3 -m http.server 8000"
echo ""
echo "Then open: http://localhost:8000/"
echo ""
echo ""
echo "5️⃣  COPY FILES FOR PAPER"
echo "━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Copy images to your paper directory:"
echo "$ cp module_ML/reports/metrics_comparison.png ~/my_paper/figures/fig1.png"
echo "$ cp module_ML/reports/confusion_matrices.png ~/my_paper/figures/fig2.png"
echo "$ cp module_ML/reports/model_summary.png ~/my_paper/figures/fig3.png"
echo ""
echo "Extract metrics for table:"
cat << 'PYEOF'
import json

report = json.load(open('module_ML/reports/arxiv_report.json'))

print("| Model | Accuracy | Macro F1 | Weighted F1 |")
print("|-------|----------|----------|-------------|")
for name, model in report['models'].items():
    m = model['metrics']
    acc = m['accuracy']
    mf1 = m['macro_f1']
    wf1 = m['weighted_f1']
    print(f"| {name:15} | {acc:.4f} | {mf1:.4f} | {wf1:.4f} |")
PYEOF
echo ""
echo ""
echo "═══════════════════════════════════════════════════════"
echo "✅ Now you can view all paper materials!"
echo ""
