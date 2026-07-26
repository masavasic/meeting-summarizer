import json
import os
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


def register_fonts() -> tuple[str, str]:
    """
    Registruje fontove koji podržavaju srpska slova.
    Na Windowsu prvo pokušava Arial, zatim Calibri.
    """
    regular_candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ]

    bold_candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
    ]

    regular_path = next(
        (
            path
            for path in regular_candidates
            if os.path.exists(path)
        ),
        None,
    )

    bold_path = next(
        (
            path
            for path in bold_candidates
            if os.path.exists(path)
        ),
        None,
    )

    if regular_path is None:
        raise FileNotFoundError(
            "Nije pronađen Arial ili Calibri font u "
            "C:\\Windows\\Fonts."
        )

    pdfmetrics.registerFont(
        TTFont(
            "ReportRegular",
            regular_path,
        )
    )

    pdfmetrics.registerFont(
        TTFont(
            "ReportBold",
            bold_path or regular_path,
        )
    )

    return "ReportRegular", "ReportBold"


def safe_text(value: Any) -> str:
    return escape(str(value).strip())


def add_list_section(
    story: list[Any],
    title: str,
    values: list[str],
    heading_style: ParagraphStyle,
    body_style: ParagraphStyle,
) -> None:
    story.append(
        Paragraph(
            safe_text(title),
            heading_style,
        )
    )

    if not values:
        story.append(
            Paragraph(
                "- Nema izdvojenih stavki.",
                body_style,
            )
        )
    else:
        for value in values:
            story.append(
                Paragraph(
                    f"- {safe_text(value)}",
                    body_style,
                )
            )

    story.append(
        Spacer(
            1,
            0.25 * cm,
        )
    )


def generate_pdf_report(
    summary_json_path: str,
    output_pdf_path: str,
) -> None:
    """
    Generiše PDF koji sadrži samo konačni sažetak.
    """

    if not os.path.exists(summary_json_path):
        raise FileNotFoundError(
            f"JSON sažetak nije pronađen: "
            f"{summary_json_path}"
        )

    with open(
        summary_json_path,
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    output_directory = os.path.dirname(
        output_pdf_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    regular_font, bold_font = register_fonts()

    document = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title=str(
            summary.get(
                "title",
                "Sažetak sadržaja",
            )
        ),
        author="Meeting Summarizer",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=styles["Normal"],
        fontName=regular_font,
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        spaceAfter=16,
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=12,
        leading=15,
        spaceBefore=8,
        spaceAfter=5,
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=10,
        leading=15,
        spaceAfter=5,
    )

    story: list[Any] = []

    story.append(
        Paragraph(
            safe_text(
                summary.get(
                    "title",
                    "Sažetak sadržaja",
                )
            ),
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Vrsta sadržaja: "
            + safe_text(
                summary.get(
                    "content_type",
                    "drugo",
                )
            ),
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            "Kratak pregled",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            safe_text(
                summary.get(
                    "short_overview",
                    "Nema dostupnog pregleda.",
                )
            ),
            body_style,
        )
    )

    story.append(
        Spacer(
            1,
            0.25 * cm,
        )
    )

    add_list_section(
        story,
        "Glavne teme",
        summary.get(
            "main_topics",
            [],
        ),
        heading_style,
        body_style,
    )

    add_list_section(
        story,
        "Važne informacije",
        summary.get(
            "important_information",
            [],
        ),
        heading_style,
        body_style,
    )

    add_list_section(
        story,
        "Brojevi i datumi",
        summary.get(
            "numbers_and_dates",
            [],
        ),
        heading_style,
        body_style,
    )

    add_list_section(
        story,
        "Zaključci",
        summary.get(
            "conclusions",
            [],
        ),
        heading_style,
        body_style,
    )

    add_list_section(
        story,
        "Akcione stavke",
        summary.get(
            "action_items",
            [],
        ),
        heading_style,
        body_style,
    )

    add_list_section(
        story,
        "Nejasni delovi transkripta",
        summary.get(
            "unclear_parts",
            [],
        ),
        heading_style,
        body_style,
    )

    document.build(story)

    print(
        f"PDF sažetak je sačuvan u: "
        f"{output_pdf_path}"
    )