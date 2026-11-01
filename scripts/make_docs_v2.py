"""Polished single-page PDF layouts for the assessment deliverables."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
ART, DOCS = ROOT / "artifacts", ROOT / "docs"
NAVY, TEAL, CYAN = map(colors.HexColor, ("#123047", "#087E8B", "#26A6B5"))
PALE, BLUE = colors.HexColor("#EAF6F7"), colors.HexColor("#EDF3F8")
INK, MUTED, LINE = map(colors.HexColor, ("#243746", "#607582", "#C5D6DE"))
GREEN, AMBER, WHITE = colors.HexColor("#278B62"), colors.HexColor("#D98E20"), colors.white


def wrap(text, font, size, width):
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if not current or stringWidth(candidate, font, size) <= width:
            current = candidate
        else:
            lines.append(current); current = word
    if current: lines.append(current)
    return lines


def block(c, text, x, y, width, font="Helvetica", size=9, leading=12, color=INK):
    c.setFont(font, size); c.setFillColor(color)
    for line in wrap(text, font, size, width):
        c.drawString(x, y, line); y -= leading
    return y


def header(c, heading, subtitle, w, h):
    c.setFillColor(NAVY); c.rect(0, h-31*mm, w, 31*mm, stroke=0, fill=1)
    c.setFillColor(WHITE); c.setFont("Helvetica-Bold", 22); c.drawString(15*mm, h-17*mm, heading)
    c.setFillColor(colors.HexColor("#D8E7EE")); c.setFont("Helvetica", 9.5); c.drawString(15*mm, h-24*mm, subtitle)


def footer(c, w, note):
    c.setStrokeColor(LINE); c.line(15*mm, 12*mm, w-15*mm, 12*mm)
    c.setFillColor(MUTED); c.setFont("Helvetica", 7.5); c.drawString(15*mm, 7.7*mm, note)
    c.drawRightString(w-15*mm, 7.7*mm, "CreditGuard AI | Credit Risk Portfolio Analytics")


def panel(c, x, y, w, h, fill=WHITE, stroke=LINE):
    c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(.7)
    c.roundRect(x, y, w, h, 4*mm, stroke=1, fill=1)


def architecture():
    w, h = landscape(A4); c = canvas.Canvas(str(DOCS/"architecture.pdf"), pagesize=(w,h))
    c.setTitle("CreditGuard AI - Architecture")
    header(c, "CreditGuard AI - Solution Architecture", "From raw portfolio data to governed analyst decisions", w, h)
    stages = [
        ("01", "Source data", "Five banking tables", ["Customers and loans", "Bureau credit lines", "Repayments and collections"], "CSV | pandas"),
        ("02", "Data foundation", "Clean account-level view", ["Validated one-to-many joins", "Quality flags and normalization", "Parquet dataset + QA report"], "pandas | PyArrow"),
        ("03", "Risk intelligence", "Scores and explanations", ["Logistic baseline", "Gradient boosting challenger", "AUC, KS, Gini + importance"], "scikit-learn"),
        ("04", "Analyst experience", "Monitor, explore, review", ["Portfolio and account views", "Filters and drill-down", "Governance metrics"], "Streamlit | Plotly"),
    ]
    margin, gap, y, ch = 14*mm, 8*mm, 102*mm, 72*mm
    cw = (w-2*margin-3*gap)/4
    for i, (num,name,promise,bullets,tools) in enumerate(stages):
        x = margin+i*(cw+gap); panel(c,x,y,cw,ch,BLUE if i%2==0 else PALE)
        c.setFillColor(TEAL); c.circle(x+10*mm,y+ch-11*mm,6*mm,stroke=0,fill=1)
        c.setFillColor(WHITE); c.setFont("Helvetica-Bold",9); c.drawCentredString(x+10*mm,y+ch-14*mm,num)
        c.setFillColor(NAVY); c.setFont("Helvetica-Bold",12); c.drawString(x+19*mm,y+ch-14*mm,name)
        block(c,promise,x+8*mm,y+ch-27*mm,cw-16*mm,"Helvetica-Bold",9,11)
        yy=y+ch-41*mm
        for bullet in bullets:
            c.setFillColor(CYAN); c.circle(x+9*mm,yy+1.5,1.2,stroke=0,fill=1)
            c.setFillColor(INK); c.setFont("Helvetica",8.2); c.drawString(x+13*mm,yy,bullet); yy-=6*mm
        c.setFillColor(NAVY); c.roundRect(x+8*mm,y+7*mm,cw-16*mm,9*mm,2.5*mm,stroke=0,fill=1)
        c.setFillColor(WHITE); c.setFont("Helvetica-Bold",7.8); c.drawCentredString(x+cw/2,y+10*mm,tools)
        if i<3:
            ax,ay=x+cw+1.5*mm,y+ch/2; c.setStrokeColor(TEAL); c.setLineWidth(2)
            c.line(ax,ay,ax+4*mm,ay); c.line(ax+4*mm,ay,ax+1.5*mm,ay+2.2*mm); c.line(ax+4*mm,ay,ax+1.5*mm,ay-2.2*mm)
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold",13); c.drawString(margin,88*mm,"Embedded AI with explicit guardrails")
    ai = [
        ("Ask the Portfolio", "Pandas calculates an approved aggregation first. OpenRouter only rewrites the verified fact.", "Question + aggregate fact only", "Deterministic fallback", TEAL),
        ("Anomaly review", "Peer comparison flags unusual behaviour and gives the analyst a plain-language reason code.", "Analyst confirms or dismisses", "No automated adverse action", GREEN),
    ]
    aw=(w-2*margin-8*mm)/2
    for i,(name,desc,left,right,accent) in enumerate(ai):
        x=margin+i*(aw+8*mm); panel(c,x,34*mm,aw,47*mm)
        c.setFillColor(accent); c.roundRect(x,34*mm,5*mm,47*mm,3*mm,stroke=0,fill=1)
        c.setFillColor(NAVY); c.setFont("Helvetica-Bold",12); c.drawString(x+11*mm,71*mm,name)
        block(c,desc,x+11*mm,61*mm,aw-19*mm,size=8.8,leading=11)
        c.setFillColor(accent); c.setFont("Helvetica-Bold",8); c.drawString(x+11*mm,43*mm,left); c.drawRightString(x+aw-8*mm,43*mm,right)
    footer(c,w,"Secrets stay in the host secret store. Raw customer rows never leave the application.")
    c.showPage(); c.save()


def solution():
    metrics=json.loads((ART/"model_metrics.json").read_text(encoding="utf-8")); data=pd.read_parquet(ART/"scored_accounts.parquet")
    selected=metrics["selected_model"]; m=metrics["models"][selected]
    loan=data.groupby("loan_type").target_default.mean().sort_values(ascending=False)
    emp=data.groupby("employment_type").target_default.mean().sort_values(ascending=False)
    region=data.groupby("region").target_default.mean().sort_values(ascending=False)
    w,h=A4; c=canvas.Canvas(str(DOCS/"solution_overview.pdf"),pagesize=A4); c.setTitle("CreditGuard AI - Solution Overview")
    header(c,"CreditGuard AI - Solution Overview","A practical decision-support workspace for portfolio risk analysts",w,h)
    margin=15*mm; full=w-2*margin
    panel(c,margin,h-66*mm,full,25*mm,PALE)
    c.setFillColor(TEAL); c.setFont("Helvetica-Bold",9); c.drawString(margin+7*mm,h-49*mm,"WHAT IT DELIVERS")
    block(c,"One workflow to understand portfolio risk, prioritize accounts, ask grounded questions and document anomaly decisions - without automating adverse action.",margin+7*mm,h-57*mm,full-14*mm,size=9.2,leading=11)
    ky,kh,kg=h-101*mm,25*mm,4*mm; kw=(full-3*kg)/4
    kpis=[(f"{len(data):,}","loan accounts"),(f"{data.target_default.mean():.1%}","observed defaults"),(f"{m['auc']:.3f}","holdout AUC"),(f"{m['ks']:.3f}","holdout KS")]
    for i,(value,label) in enumerate(kpis):
        x=margin+i*(kw+kg); panel(c,x,ky,kw,kh)
        c.setFillColor(NAVY); c.setFont("Helvetica-Bold",17); c.drawCentredString(x+kw/2,ky+12*mm,value)
        c.setFillColor(MUTED); c.setFont("Helvetica",7.8); c.drawCentredString(x+kw/2,ky+5.5*mm,label)
    gap=7*mm; col=(full-gap)/2; by=h-165*mm; bh=55*mm
    panel(c,margin,by,col,bh,BLUE); panel(c,margin+col+gap,by,col,bh,PALE)
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold",11.5); c.drawString(margin+7*mm,by+bh-10*mm,"How an analyst uses it")
    yy=by+bh-21*mm
    for item in ["Scan exposure, defaults and risk segments","Filter by product, region or employment","Open an account and review its drivers","Investigate and record anomaly decisions"]:
        c.setFillColor(TEAL); c.circle(margin+8*mm,yy+1.5,1.2,stroke=0,fill=1); c.setFillColor(INK); c.setFont("Helvetica",8.3); c.drawString(margin+12*mm,yy,item); yy-=8*mm
    rx=margin+col+gap; c.setFillColor(NAVY); c.setFont("Helvetica-Bold",11.5); c.drawString(rx+7*mm,by+bh-10*mm,"AI features with human control")
    c.setFillColor(TEAL); c.setFont("Helvetica-Bold",8.7); c.drawString(rx+7*mm,by+bh-22*mm,"ASK THE PORTFOLIO")
    block(c,"Verified aggregation first; OpenRouter wording second. Unsupported questions are declined.",rx+7*mm,by+bh-29*mm,col-14*mm,size=8,leading=9.5)
    c.setFillColor(GREEN); c.setFont("Helvetica-Bold",8.7); c.drawString(rx+7*mm,by+17*mm,"ANOMALY REVIEW")
    block(c,"Peer-relative flags include reason codes. The analyst confirms or dismisses every case.",rx+7*mm,by+10*mm,col-14*mm,size=8,leading=9.5)
    iy,ih=h-230*mm,56*mm; panel(c,margin,iy,full,ih)
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold",11.5); c.drawString(margin+7*mm,iy+ih-10*mm,"Three portfolio signals - and why they matter")
    insights=[(f"{loan.index[0]} | {loan.iloc[0]:.1%} default","Prioritize product-level underwriting and collections review."),(f"{emp.index[0]} | {emp.iloc[0]:.1%} default","Validate income stability; avoid blanket segment exclusions."),(f"{region.index[0]} | {region.iloc[0]:.1%} default","Check product and employment mix before changing limits.")]
    iw=(full-14*mm)/3
    for i,(signal,meaning) in enumerate(insights):
        x=margin+7*mm+i*iw; offset=4*mm if i else 0
        if i: c.setStrokeColor(LINE); c.line(x,iy+8*mm,x,iy+ih-17*mm)
        c.setFillColor(AMBER); c.setFont("Helvetica-Bold",8.7); c.drawString(x+offset,iy+30*mm,signal)
        block(c,meaning,x+offset,iy+22*mm,iw-8*mm,size=7.8,leading=9)
    py,ph=24*mm,34*mm; panel(c,margin,py,full,ph,NAVY,NAVY)
    c.setFillColor(WHITE); c.setFont("Helvetica-Bold",10); c.drawString(margin+7*mm,py+ph-9*mm,"Production path")
    block(c,"Add point-in-time controls, durable review storage, access management, model registry, calibration and drift monitoring, fairness testing, audit logs and bank-approved OpenRouter data-processing terms.",margin+7*mm,py+ph-17*mm,full-14*mm,size=8,leading=10,color=colors.HexColor("#DCE8ED"))
    footer(c,w,f"Selected model: {selected.replace('_',' ')} | Gini {m['gini']:.3f} | Decision support only")
    c.showPage(); c.save()


if __name__ == "__main__":
    architecture(); solution(); print("Created polished one-page PDFs")
