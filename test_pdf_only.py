from processing.pdf_report import generate_pdf_report

generate_pdf_report(
    summary_json_path="outputs/summary_test.json",
    output_pdf_path="outputs/meeting_report_test.pdf",
)

print("PDF napravljen: outputs/meeting_report_test.pdf")