"""Build the paginated technical report from saved experiment records."""
import json
from datetime import datetime
from html import escape
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, Preformatted

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"


def main():
    results = json.loads((OUT/"results.json").read_text())
    audit = json.loads((OUT/"data_audit.json").read_text())
    replay = json.loads((OUT/"evaluation_verification.json").read_text())
    live = json.loads((OUT/"live_browser_verification.json").read_text())
    choices = json.loads((OUT/"validation_selection.json").read_text())
    errors = json.loads((OUT/"misclassifications.json").read_text())
    selected = results["test_results"]["resnet18_frozen_logistic"]
    baseline = results["test_results"]["pixels_logistic"]
    score = selected["metrics"]
    date = datetime.fromisoformat(replay["verified_at_utc"].replace("Z", "+00:00")).strftime("%d %B %Y")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle",fontName="Helvetica-Bold",fontSize=23,leading=28,spaceAfter=10,textColor=colors.HexColor("#173e46")))
    styles.add(ParagraphStyle(name="ReportSub",fontName="Helvetica",fontSize=10,leading=14,spaceAfter=15,textColor=colors.HexColor("#53616a")))
    styles.add(ParagraphStyle(name="ReportBody",fontName="Helvetica",fontSize=9.3,leading=13.2,spaceAfter=8))
    styles.add(ParagraphStyle(name="ReportSmall",fontName="Helvetica",fontSize=7.8,leading=10.5,spaceAfter=6))
    styles.add(ParagraphStyle(name="ReportHeading",fontName="Helvetica-Bold",fontSize=12.5,leading=16,spaceBefore=11,spaceAfter=8,textColor=colors.HexColor("#173e46")))
    styles.add(ParagraphStyle(name="ReportCaption",fontName="Helvetica-Oblique",fontSize=8,leading=11,alignment=TA_CENTER,spaceAfter=10,textColor=colors.HexColor("#53616a")))
    story=[]
    def paragraph(text,style="ReportBody"):
        story.append(Paragraph(escape(text),styles[style]))
    def heading(text): paragraph(text,"ReportHeading")
    def table(rows,widths):
        cells=[[Paragraph(escape(str(value)),styles["ReportSmall"]) for value in row] for row in rows]
        element=Table(cells,colWidths=widths,repeatRows=1,hAlign="LEFT")
        element.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e4edf0")),("VALIGN",(0,0),(-1,-1),"TOP"),("LINEBELOW",(0,0),(-1,0),0.6,colors.HexColor("#98aeb5")),("LINEBELOW",(0,1),(-1,-1),0.3,colors.HexColor("#dce3e6")),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        story.extend([element,Spacer(1,9)])
    def figure(name,width,caption):
        from PIL import Image as PillowImage
        with PillowImage.open(OUT/name) as im: ratio=im.height/im.width
        story.append(Image(str(OUT/name),width=width,height=width*ratio))
        paragraph(caption,"ReportCaption")
    def pct(value):return f"{value*100:.2f}%"
    paragraph("Casting Defect Classification","ReportTitle")
    paragraph(f"Technical report | {date}","ReportSub")
    paragraph(f"A CPU-trained binary classifier identifies defective front-view castings. A frozen ImageNet ResNet-18 feature extractor with a trained logistic head achieves {pct(score['accuracy'])} accuracy and {pct(score['defect_recall'])} defect recall on 715 held-out images. The test set contains two missed defects and three false alarms. The project includes a simpler baseline, validation-based tuning, error analysis, public image-upload inference, and a containerized REST API.")
    heading("1. Problem and dataset")
    paragraph("Image-level folder labels support binary classification: ok_front = 0 and def_front = 1. The public industrial dataset contains 7,348 images at 300 x 300 pixels. Missed defects are the main inspection risk; false alarms create additional manual inspection.")
    rows=[["Partition","Acceptable","Defective","Total"],["Supplied training",2875,3758,6633]]
    for key,label in [("train","Final training"),("validation","Validation"),("test","Supplied test")]:
        dist=audit["split_distribution"][key];rows.append([label,dist["ok_front"],dist["def_front"],sum(dist.values())])
    table(rows,[6.2*cm,3.3*cm,3.3*cm,3.3*cm])
    paragraph("The audit found 64 training images identical to test images. Their training copies were removed before a seeded, stratified validation split; all 715 test images were retained. No corrupt files or conflicting exact-duplicate labels were found. Final splits have disjoint decoded-pixel hashes. The related 512 x 512 collection is excluded. Missing physical-part/session identifiers prevent ruling out transformed near duplicates.")
    heading("2. Models and training")
    paragraph("The baseline uses standardized grayscale 32 x 32 pixels and class-weighted logistic regression. The stronger model uses RGB 128 x 128 images, antialiased bilinear resizing, ImageNet normalization, and frozen ResNet-18 features. Its logistic head is trained on original images plus one seeded flipped/brightness-adjusted view per training image. The encoder is not fine-tuned.")
    paragraph(f"Four regularization values per model are compared on validation data. Thresholds maximize validation defect F2; macro F1 selects regularization and model family. Selected C/threshold values are {choices['baseline']['C']:g}/{baseline['threshold']:.2f} for the baseline and {choices['transfer']['C']:g}/{selected['threshold']:.2f} for ResNet. Test results do not influence selection. Class weights address moderate imbalance. The initial full pipeline took {results['training_seconds']:.1f} seconds with four CPU threads and cached pretrained weights; seed = 42.")
    story.append(PageBreak())
    heading("3. Evaluation results")
    table([["Metric","Pixel baseline","ResNet + logistic head"],*[ [label,fmt(baseline["metrics"][key]),fmt(score[key])] for key,label,fmt in [("accuracy","Accuracy",pct),("balanced_accuracy","Balanced accuracy",pct),("macro_f1","Macro F1",lambda x:f"{x:.4f}"),("defect_precision","Defect precision",pct),("defect_recall","Defect recall",pct),("defect_f2","Defect F2",lambda x:f"{x:.4f}"),("roc_auc","ROC AUC",lambda x:f"{x:.4f}"),("defect_average_precision","Defect average precision",lambda x:f"{x:.4f}"),("brier_score","Brier score",lambda x:f"{x:.5f}")]]],[6.2*cm,4.95*cm,4.95*cm])
    paragraph("The selected model correctly predicts 710 of 715 images: 259 acceptable and 451 defective. Its confusion matrix is [[259, 3], [2, 451]], with true labels as rows and acceptable first. The baseline produces 26 missed defects and 76 false alarms.")
    ci=selected["ci95_stratified_bootstrap"]
    paragraph(f"Stratified bootstrap with 1,000 resamples gives 95% intervals of {pct(ci['accuracy'][0])} to {pct(ci['accuracy'][1])} for accuracy, {pct(ci['defect_recall'][0])} to {pct(ci['defect_recall'][1])} for defect recall, and {ci['macro_f1'][0]:.4f} to {ci['macro_f1'][1]:.4f} for macro F1. These intervals assume independent observations within class; related acquisitions could make them too narrow. Probabilities are uncalibrated.")
    figure("confusion_matrix.png",9.2*cm,"Figure 1. Selected model: held-out test confusion matrix.")
    story.append(PageBreak())
    heading("4. Error analysis")
    paragraph("All five errors are shown below. Visual observations are hypotheses, not expert defect diagnoses. Labels and decision thresholds remain unchanged after test inspection.")
    figure("error_gallery.png",16.1*cm,"Figure 2. All five misclassified test images; 1 = defective, 0 = acceptable.")
    observations={"cast_def_0_2154.jpeg":"Missed defect. A small inner-rim irregularity may be lost at low resolution; the low score also limits uncertainty-only screening.","cast_def_0_1607.jpeg":"Missed defect. Small rim/contour irregularities under strong illumination merit higher-resolution comparison.","cast_ok_0_259.jpeg":"False alarm. Surface texture and diagonal illumination may resemble defect cues.","cast_ok_0_1137.jpeg":"False alarm. Uneven shading and surface marks accompany a score close to the threshold.","cast_ok_0_7839.jpeg":"False alarm. Fine machining texture and pose/lighting differences accompany a borderline positive score."}
    table([["Image","Defect score","Observation"],*[[Path(row["path"]).name,f"{row['defect_probability']:.3f}",observations[Path(row["path"]).name]] for row in errors]],[4.2*cm,2.0*cm,9.9*cm])
    paragraph("Four errors occur in the middle contrast third (4/238); one occurs in the low third (1/238), and none in the high third (0/239). Both missed defects occur in the middle brightness third. These descriptive slices do not support a simple extreme-lighting explanation and are too small to establish a stable causal pattern.")
    story.append(PageBreak())
    heading("5. Deployment and verification")
    paragraph("GitHub Pages serves an image-upload interface using ONNX Runtime Web and the exported trained model. RGB conversion and antialiased 128 x 128 resizing precede inference; normalization and the fused scaler/head are inside the graph. Images remain in the browser. Vendored runtime assets eliminate a CDN dependency. Initial model/runtime downloads and browser CPU/memory use are the main deployment costs.")
    paragraph("The optional FastAPI service exposes /health and /predict, validates PNG/JPEG uploads, and enforces 10 MiB/20 megapixel limits. Startup verifies the model checksum and performs warm-up. The Docker image runs as a non-root user with a health check and runtime-only dependencies. Authentication and global rate limiting are outside the prototype scope.")
    table([["Check","Recorded result"],["Automated tests","16 passed locally; repository automated tests passed."],["Public inference","Both classes passed example buttons and actual file uploads; invalid images rejected; no browser/network/HTTP errors."],["Artifacts and data",f"All {replay['dataset_files_verified']:,} source hashes match the manifest. Model hashes match committed artifacts. All {replay['test_images_replayed']} test images reproduce the original confusion matrices."],["Metrics and numerical parity","Saved metrics agree within 1e-8; bootstrap intervals reproduce. Maximum ONNX/saved score difference: 7.57e-6; Torch/ONNX: 7.99e-6."],["Verification scope","Saved-artifact inference replay; no retraining or clean-environment installation during verification."]],[4.2*cm,11.9*cm])
    paragraph("Reproduction uses Python 3.12, pinned direct dependencies, config.json, and an external dataset in train/test class folders. The report is generated from saved experiment records. Commands from the repository root:")
    story.append(Preformatted("python -m pip install -r requirements-dev.txt\npython -m pytest -q\npython -m scripts.verify_evaluation --dataset-root DATASET_ROOT\npython -m scripts.generate_report",ParagraphStyle(name="Commands",fontName="Courier",fontSize=8,leading=11,spaceAfter=10)))
    heading("6. Limitations and further experiments")
    paragraph("The collection covers one part/view and restricted acquisition conditions, without defect-subtype or location annotations. Future experiments include part/session-held-out evaluation, expert-confirmed subtle defects and hard negatives, higher-resolution inputs, augmentation ablations, final-block fine-tuning, probability calibration, and lighting/camera shift evaluation. Detection or segmentation requires location annotations. No test-derived hypothesis has been used to retune the reported model.")
    heading("References")
    for label,url in [("Dataset: Ravirajsinh Dabhi, casting product image data for quality inspection","https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product"),("Source code and machine-readable evaluation records","https://github.com/Abdullah-Sarwar-MIrza/casting-defect-ml-assessment"),("Public image-upload inference","https://abdullah-sarwar-mirza.github.io/casting-defect-ml-assessment/")]:
        story.append(Paragraph(f'<link href="{escape(url)}" color="#176254">{escape(label)}</link>',styles["ReportSmall"]))
    def furniture(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor("#c4d2d7"));canvas.line(1.8*cm,A4[1]-1.3*cm,A4[0]-1.8*cm,A4[1]-1.3*cm)
        canvas.setFont("Helvetica",7.5);canvas.setFillColor(colors.HexColor("#53616a"));canvas.drawString(1.8*cm,A4[1]-1.05*cm,"CASTING DEFECT CLASSIFICATION | TECHNICAL REPORT")
        canvas.drawString(1.8*cm,1.0*cm,date);canvas.drawRightString(A4[0]-1.8*cm,1.0*cm,f"Page {doc.page}");canvas.restoreState()
    doc=SimpleDocTemplate(str(OUT/"technical_report.pdf"),pagesize=A4,rightMargin=1.8*cm,leftMargin=1.8*cm,topMargin=1.75*cm,bottomMargin=1.6*cm,title="Casting Defect Classification - Technical Report",author="",subject="Dataset preparation, model selection, held-out evaluation and inference")
    doc.build(story,onFirstPage=furniture,onLaterPages=furniture)
    print(OUT/"technical_report.pdf")


if __name__ == "__main__":
    main()
