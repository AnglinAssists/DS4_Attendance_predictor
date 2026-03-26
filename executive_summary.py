#!/usr/bin/env python3
"""
I generate an executive summary PDF from the analysis results.
This is the deliverable I'd hand to a principal, program director, or
anyone who needs the findings without running Python code.

I switched from fpdf2 to reportlab because fpdf2 kept clipping text
on the left margin. reportlab's Platypus layout engine handles margins
and centering properly every time.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak,
)
from reportlab.platypus.flowables import HRFlowable

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# I define colors once so the PDF has a consistent look.
NAVY = HexColor("#1a5276")
GRAY = HexColor("#5d6d7e")
DARK = HexColor("#333333")
BODY_COLOR = HexColor("#3a3a3a")
LIGHT_BG = HexColor("#f0f8ff")
GREEN_BG = HexColor("#f5faf5")
GREEN_TEXT = HexColor("#27ae60")
MUTED = HexColor("#787878")


def _styles():
    """I build all the paragraph styles I need in one place."""
    return {
        "title": ParagraphStyle(
            "title", fontName="Helvetica-Bold", fontSize=22,
            leading=26, alignment=TA_CENTER, textColor=NAVY,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontName="Helvetica", fontSize=10,
            leading=14, alignment=TA_CENTER, textColor=GRAY,
            spaceAfter=16,
        ),
        "heading": ParagraphStyle(
            "heading", fontName="Helvetica-Bold", fontSize=11,
            leading=15, textColor=NAVY, spaceBefore=12, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=9, leading=13,
            textColor=BODY_COLOR, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small", fontName="Helvetica", fontSize=8, leading=11,
            textColor=BODY_COLOR,
        ),
        "small_bold": ParagraphStyle(
            "small_bold", fontName="Helvetica-Bold", fontSize=8,
            leading=11, textColor=NAVY,
        ),
        "scorecard_label": ParagraphStyle(
            "sc_label", fontName="Helvetica", fontSize=8, leading=11,
            textColor=DARK,
        ),
        "scorecard_yes": ParagraphStyle(
            "sc_yes", fontName="Helvetica-Bold", fontSize=8,
            leading=11, textColor=GREEN_TEXT,
        ),
        "scorecard_detail": ParagraphStyle(
            "sc_detail", fontName="Helvetica-Oblique", fontSize=7.5,
            leading=10, textColor=MUTED,
        ),
        "verdict": ParagraphStyle(
            "verdict", fontName="Helvetica-Bold", fontSize=9,
            leading=13, textColor=NAVY, spaceBefore=6, spaceAfter=10,
        ),
        "rec_num": ParagraphStyle(
            "rec_num", fontName="Helvetica-Bold", fontSize=9,
            leading=13, textColor=BODY_COLOR,
        ),
        "rec_text": ParagraphStyle(
            "rec_text", fontName="Helvetica", fontSize=9,
            leading=13, textColor=BODY_COLOR,
        ),
        "footnote": ParagraphStyle(
            "footnote", fontName="Helvetica-Oblique", fontSize=7.5,
            leading=10, textColor=MUTED, spaceBefore=14,
        ),
        "card_num": ParagraphStyle(
            "card_num", fontName="Helvetica-Bold", fontSize=16,
            leading=20, alignment=TA_CENTER, textColor=NAVY,
        ),
        "card_label": ParagraphStyle(
            "card_label", fontName="Helvetica", fontSize=7.5,
            leading=10, alignment=TA_CENTER, textColor=GRAY,
        ),
    }


def _header_footer(canvas, doc):
    """I draw the header and footer directly on the canvas for every page."""
    canvas.saveState()
    w, h = letter

    # Header text
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.setFillColor(GRAY)
    canvas.drawCentredString(
        w / 2, h - 36,
        "CSD 4 K-4 Attendance Analysis  |  East Harlem  |  February 2026"
    )

    # Header line
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(0.5)
    canvas.line(54, h - 42, w - 54, h - 42)

    # Footer
    canvas.setFont("Helvetica-Oblique", 7)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(
        w / 2, 28,
        "Mark Anglin  |  Reading Partners NYC  |  github.com/AnglinAssists"
    )

    canvas.restoreState()


def build_summary():
    out_path = os.path.join(OUTPUT_DIR, "executive_summary.pdf")
    doc = SimpleDocTemplate(
        out_path,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.6 * inch,
    )

    s = _styles()
    story = []
    avail_w = letter[0] - 1.5 * inch  # usable width inside margins

    # --- TITLE ---
    story.append(Spacer(1, 16))
    story.append(Paragraph("Blizzard Impact on School Attendance", s["title"]))
    story.append(Paragraph(
        "CSD 4 (East Harlem)  |  Grades K-4  |  Prediction for February 24, 2026",
        s["subtitle"],
    ))

    # --- CONTEXT ---
    story.append(Paragraph("Context", s["heading"]))
    story.append(Paragraph(
        "On February 22, 2026, the Blizzard of 2026 dropped 22 inches of snow on NYC. "
        "Schools closed Monday the 23rd and reopened Tuesday the 24th with MTA service "
        "running at reduced capacity. CSD 4 serves 3,196 K-4 students across 18 "
        "elementary schools in East Harlem, where 86% of students are economically "
        "disadvantaged. This analysis uses 84 days of attendance and weather data "
        "to predict how many students would attend on the return day.",
        s["body"],
    ))

    # --- KEY FINDINGS: four stat cards ---
    story.append(Paragraph("Key Findings", s["heading"]))
    story.append(Spacer(1, 4))

    cards = [
        ("75.0%", "Predicted\nAttendance"),
        ("2,397", "Students\nExpected"),
        ("~415", "Extra\nAbsences"),
        ("72-82%", "95% Confidence\nInterval"),
    ]
    card_w = avail_w / 4

    card_data = [[]]
    for big_num, label in cards:
        cell_content = [
            Paragraph(big_num, s["card_num"]),
            Spacer(1, 2),
            Paragraph(label.replace("\n", "<br/>"), s["card_label"]),
        ]
        card_data[0].append(cell_content)

    card_table = Table(card_data, colWidths=[card_w] * 4)
    card_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("BOX", (0, 0), (0, 0), 0.5, LIGHT_BG),
        ("BOX", (1, 0), (1, 0), 0.5, LIGHT_BG),
        ("BOX", (2, 0), (2, 0), 0.5, LIGHT_BG),
        ("BOX", (3, 0), (3, 0), 0.5, LIGHT_BG),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 8))

    # --- HYPOTHESIS TESTING ---
    story.append(Paragraph("Hypothesis Testing: Prediction vs Actual", s["heading"]))
    story.append(Paragraph(
        "Actual Feb 24 attendance was 79.3% (2,536 students). The model predicted "
        "75.0%. The actual value falls within the 95% confidence interval "
        "[72.2%, 82.1%], confirming the model was well-calibrated.",
        s["body"],
    ))

    # Scorecard table
    scorecard_rows = [
        [Paragraph("<b>Scorecard</b>", s["small_bold"]), "", ""],
        [
            Paragraph("Blizzard effect is real?", s["scorecard_label"]),
            Paragraph("YES", s["scorecard_yes"]),
            Paragraph("Paired t-test p &lt; 0.0001", s["scorecard_detail"]),
        ],
        [
            Paragraph("Effect size is meaningful?", s["scorecard_label"]),
            Paragraph("YES", s["scorecard_yes"]),
            Paragraph("Cohen's d = -3.11 (large)", s["scorecard_detail"]),
        ],
        [
            Paragraph("Prediction within 95% CI?", s["scorecard_label"]),
            Paragraph("YES", s["scorecard_yes"]),
            Paragraph("Actual 79.3% in [72.2%, 82.1%]", s["scorecard_detail"]),
        ],
        [
            Paragraph("Model statistically accurate?", s["scorecard_label"]),
            Paragraph("YES", s["scorecard_yes"]),
            Paragraph("Bootstrap z-test p = 0.35", s["scorecard_detail"]),
        ],
    ]
    sc_table = Table(
        scorecard_rows,
        colWidths=[2.6 * inch, 0.5 * inch, avail_w - 3.1 * inch],
    )
    sc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GREEN_BG),
        ("SPAN", (0, 0), (-1, 0)),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(sc_table)
    story.append(Paragraph(
        "Verdict: STRONG SUPPORT - the blizzard effect is real, "
        "and the model is operationally useful.",
        s["verdict"],
    ))

    # --- FIGURE 7 ---
    fig7_path = os.path.join(OUTPUT_DIR, "fig7_hypothesis_tests.png")
    if os.path.exists(fig7_path):
        story.append(Paragraph("Prediction vs Actual", s["heading"]))
        story.append(Image(fig7_path, width=avail_w, height=avail_w * 0.45))
        story.append(Spacer(1, 6))

    # --- FIGURE 1 ---
    fig1_path = os.path.join(OUTPUT_DIR, "fig1_timeseries.png")
    if os.path.exists(fig1_path):
        story.append(Paragraph("Attendance Over Time", s["heading"]))
        story.append(Image(fig1_path, width=avail_w, height=avail_w * 0.45))
        story.append(Spacer(1, 6))

    # --- RECOMMENDATIONS ---
    story.append(Paragraph("Recommendations", s["heading"]))
    recs = [
        "Pre-position tutoring and program staff based on 70-80% attendance "
        "forecast before reopening.",
        "Target outreach to chronically absent families the evening before "
        "return day.",
        "Delay assessments by one day to avoid mass makeups and scheduling "
        "disruption.",
        "Share attendance forecasts with meal programs so they can adjust "
        "food prep quantities.",
        "Run this model weekly using weather forecasts to flag any sub-80% "
        "attendance days in advance.",
    ]
    rec_data = []
    for i, rec in enumerate(recs, 1):
        rec_data.append([
            Paragraph(f"<b>{i}.</b>", s["rec_num"]),
            Paragraph(rec, s["rec_text"]),
        ])
    rec_table = Table(rec_data, colWidths=[0.3 * inch, avail_w - 0.3 * inch])
    rec_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(rec_table)

    # --- METHODOLOGY FOOTNOTE ---
    story.append(Paragraph(
        "<i>Methodology: Weighted ensemble of OLS regression, Random Forest (n=200), "
        "and Gradient Boosting (n=200) trained on 83 school days of attendance and "
        "NOAA weather data. 95% CI via 500-iteration bootstrap. Data sources: NYC DOE "
        "attendance records, NYSED 2023-24 enrollment, NOAA Central Park observations.</i>",
        s["footnote"],
    ))

    # Build the PDF
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    print(f"Executive summary saved to: {out_path}")


if __name__ == "__main__":
    build_summary()
