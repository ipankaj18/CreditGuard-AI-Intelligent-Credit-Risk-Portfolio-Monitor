from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.2, leading=10.2, textColor=colors.HexColor("#243247")))
styles.add(ParagraphStyle(name="Tiny", parent=styles["BodyText"], fontSize=7.2, leading=8.7, textColor=colors.HexColor("#243247")))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=11, leading=13, textColor=colors.HexColor("#0B5A70"), spaceBefore=5, spaceAfter=3))


def pdf(path, pagesize=A4):
    return SimpleDocTemplate(str(path), pagesize=pagesize, rightMargin=12*mm, leftMargin=12*mm, topMargin=11*mm, bottomMargin=10*mm, title=path.stem)


def architecture():
    story = [Paragraph("Credit Risk Portfolio Analytics — Architecture", styles["Title"]), Spacer(1, 4*mm)]
    cells = [
        ("1 · Raw data", "5 synthetic CSVs<br/>customers · loans · bureau · repayments · collections<br/><b>Tool:</b> pandas"),
        ("2 · Ingestion", "Schema checks · normalization · missingness flags<br/>one-to-many aggregation to account grain<br/><b>Output:</b> analysis_dataset.parquet"),
        ("3 · Analytics & model", "Portfolio cuts · engineered repayment/bureau features<br/>logistic baseline + histogram gradient boosting<br/><b>Tools:</b> pandas · scikit-learn"),
        ("4 · Product", "Streamlit portfolio/account views · Plotly charts<br/>risk score, drivers, filters and review workflows<br/><b>Hosting:</b> Streamlit Community Cloud"),
    ]
    table_data = [[Paragraph(f"<b>{a}</b><br/>{b}", styles["Small"]) for a,b in cells]]
    t = Table(table_data, colWidths=[68*mm]*4, rowHeights=[42*mm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#EAF3F5")),("BOX",(0,0),(-1,-1),1,colors.HexColor("#0B5A70")),("INNERGRID",(0,0),(-1,-1),.5,colors.white),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),7)]))
    story += [t, Spacer(1, 5*mm), Paragraph("Embedded AI controls", styles["H2x"])]
    ai = [
        [Paragraph("<b>Ask the Portfolio</b>", styles["Small"]), Paragraph("Question → allow-listed intent/dimension → pandas aggregation over active filters → verified fact → OpenRouter free-model call for wording only. OpenRouter receives the question and computed fact, never raw customer rows. Unsupported questions are declined; deterministic fallback remains available without a key or during rate limits.", styles["Small"])],
        [Paragraph("<b>Anomaly detection</b>", styles["Small"]), Paragraph("Account features → loan-type × income-band peer cohort → median/MAD robust score → reason code → analyst Confirm/Dismiss. No LLM and no automated adverse action. Review state should move to an authenticated database in production.", styles["Small"])],
    ]
    at = Table(ai, colWidths=[45*mm, 225*mm])
    at.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.5,colors.HexColor("#9AB8C1")),("VALIGN",(0,0),(-1,-1),"TOP"),("BACKGROUND",(0,0),(0,-1),colors.HexColor("#D9ECEF")),("PADDING",(0,0),(-1,-1),6)]))
    story += [at, Spacer(1, 4*mm), Paragraph("Artifacts & governance", styles["H2x"]), Paragraph("Artifacts: clean/scored Parquet, fitted joblib pipeline, metrics JSON, data-quality report. Secrets: OPENROUTER_API_KEY only in the host secret store. Monitoring: data quality → drift/calibration/discrimination → review outcomes → controlled retraining and approval. Raw IDs remain in-app; OpenRouter receives only aggregate facts.", styles["Small"])]
    pdf(DOCS / "architecture.pdf", landscape(A4)).build(story)


def solution():
    metrics = json.loads((ART / "model_metrics.json").read_text())
    data = pd.read_parquet(ART / "scored_accounts.parquet")
    selected = metrics["selected_model"]
    m = metrics["models"][selected]
    by_loan = data.groupby("loan_type").target_default.mean().sort_values(ascending=False)
    by_emp = data.groupby("employment_type").target_default.mean().sort_values(ascending=False)
    by_region = data.groupby("region").target_default.mean().sort_values(ascending=False)
    story = [Paragraph("Credit Risk Portfolio Analytics — Solution Overview", styles["Title"])]
    sections = [
        ("Analyst workflow", "Portfolio view tracks exposure, defaults and predicted risk across loan, region, employment and cross-dimensional cuts. Filters flow into all views. Account review shows score, segment and observed drivers. Model governance shows holdout performance, global importance and monitoring controls."),
        ("AI features & controls", "<b>Ask:</b> submitting a supported question triggers a computed aggregation; the optional LLM only phrases that verified fact. FP: misunderstood filter/denominator; FN: valid but unsupported wording; analyst checks source and filters. <b>Anomaly:</b> ≥3 robust SD above loan/income peers triggers a reason-coded flag. FP: legitimate unusual customer; FN: deterioration within peer norms; analyst confirms/dismisses, with no automatic adverse action."),
        ("Model", f"Account-grain bureau, repayment, collections, customer and loan features; stratified 75/25 holdout; median/mode imputation fitted inside pipelines; logistic baseline and histogram gradient-boosting challenger. Selected: <b>{selected.replace('_',' ')}</b>. Holdout AUC <b>{m['auc']:.3f}</b>, KS <b>{m['ks']:.3f}</b>, Gini <b>{m['gini']:.3f}</b>. Monitor AUC/KS/Gini, calibration, PSI, missingness, fairness and review outcomes monthly."),
        ("Top portfolio insights", f"1) {by_loan.index[0]} has the highest default rate ({by_loan.iloc[0]:.1%}); prioritize underwriting/collections review for that product. 2) {by_emp.index[0]} customers have the highest employment-segment default rate ({by_emp.iloc[0]:.1%}); verify income stability and avoid blanket policy decisions. 3) {by_region.index[0]} has the highest regional default rate ({by_region.iloc[0]:.1%}); investigate product/employment mix before changing limits."),
        ("Assumptions & production needs", "Synthetic target and histories are treated as observation-safe; scores are decision support, not adverse-action automation. Production needs: governed source contracts and lineage, point-in-time feature store, identity/access controls, encrypted secrets, model registry/validation, audit logs and durable review database, consent/retention rules, bias and regulatory review, monitoring SLAs, incident rollback and bank-approved LLM/data-processing terms."),
    ]
    for title, text in sections:
        story += [Paragraph(title, styles["H2x"]), Paragraph(text, styles["Small"])]
    pdf(DOCS / "solution_overview.pdf").build(story)


if __name__ == "__main__":
    from make_docs_v2 import architecture as architecture_v2, solution as solution_v2

    architecture_v2()
    solution_v2()
    print("Created polished one-page architecture and solution overview PDFs")
