from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

model = YOLO('results/yolov8n.pt')
res = model.val(data=r"C:\Users\arindam\Documents\BRSDD\data.yaml",
                split='val', batch=16, imgsz=640, save=True, plots=True)

# mean_results -> (precision, recall, mAP@0.5, mAP@0.5:0.95)
prec, rec, map50, map5095 = res.mean_results()

# F1 = 2 * (P * R) / (P + R)
f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0

print(f"Precision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"F1 score:  {f1*100:.2f}%")         # use this as a single "accuracy-like" percentage
print(f"mAP@0.5:   {map50*100:.2f}%")      # also a very useful single-number metric
print(f"mAP@0.5:0.95: {map5095*100:.2f}%")

# Plot confusion matrix
# Get confusion matrix data
cm = res.confusion_matrix.matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot with annotations
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
            xticklabels=res.confusion_matrix.names, 
            yticklabels=res.confusion_matrix.names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot training curves
df = pd.read_csv('results.csv')
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Curves')
plt.legend()
plt.grid(True)
plt.savefig('training_curves.png')
plt.close()

# Learning curve (losses)
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
plt.plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
plt.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
plt.plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.savefig('learning_curves.png')
plt.close()

# For ROC curve, in object detection, we can plot PR curve
# Assuming res has pr_curve
try:
    res.pr_curve.plot()
    plt.title('Precision-Recall Curve')
    plt.savefig('pr_curve.png')
    plt.close()
except:
    print("PR curve not available")
