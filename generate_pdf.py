import markdown
import subprocess
from pathlib import Path

# Paths
root = Path(__file__).resolve().parent
md_path = root / "reports" / "final_evaluation_report.md"
html_path = root / "reports" / "final_evaluation_report.html"
pdf_path = root / "reports" / "final_evaluation_report.pdf"

with open(md_path, "r", encoding="utf-8") as f:
    md_content = f.read()

# Convert to HTML
html_body = markdown.markdown(md_content, extensions=["tables", "fenced_code"])

# Professional Executive Styling
html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CrowdWisdomTrading - Final Evaluation Report</title>
    <style>
        @page {{
            size: A4;
            margin: 18mm 16mm 18mm 16mm;
            @bottom-right {{
                content: counter(page);
            }}
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #1e293b;
            line-height: 1.55;
            font-size: 10.5pt;
            background: #ffffff;
            margin: 0;
            padding: 0;
        }}
        h1 {{
            color: #0f172a;
            font-size: 20pt;
            font-weight: 800;
            border-bottom: 2px solid #f7931a;
            padding-bottom: 8px;
            margin-top: 0;
            margin-bottom: 4px;
            letter-spacing: -0.02em;
        }}
        h2 {{
            color: #1e293b;
            font-size: 13pt;
            font-weight: 700;
            margin-top: 20pt;
            margin-bottom: 8pt;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4pt;
            page-break-after: avoid;
        }}
        h3 {{
            color: #334155;
            font-size: 11pt;
            font-weight: 600;
            margin-top: 14pt;
            margin-bottom: 6pt;
            page-break-after: avoid;
        }}
        p, li {{
            color: #334155;
            font-size: 10pt;
            margin-top: 4pt;
            margin-bottom: 6pt;
        }}
        strong {{
            color: #0f172a;
            font-weight: 600;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 8pt;
            margin-bottom: 14pt;
            font-size: 9pt;
            page-break-inside: avoid;
        }}
        th {{
            background-color: #0f172a;
            color: #ffffff;
            font-weight: 600;
            text-align: left;
            padding: 7pt 9pt;
            border: 1px solid #0f172a;
        }}
        td {{
            padding: 6pt 9pt;
            border: 1px solid #cbd5e1;
            color: #1e293b;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        pre, code {{
            font-family: "JetBrains Mono", Consolas, "Courier New", monospace;
            font-size: 8.5pt;
        }}
        pre {{
            background: #0f172a;
            color: #f1f5f9;
            padding: 10pt 12pt;
            border-radius: 6pt;
            overflow-x: auto;
            margin: 8pt 0 12pt 0;
            line-height: 1.35;
            page-break-inside: avoid;
        }}
        p code, li code, td code {{
            background: #f1f5f9;
            color: #0f172a;
            padding: 1pt 4pt;
            border-radius: 3pt;
            border: 1px solid #e2e8f0;
            font-size: 8.5pt;
        }}
        blockquote {{
            margin: 10pt 0;
            padding: 8pt 14pt;
            background: #f8fafc;
            border-left: 4px solid #f7931a;
            color: #334155;
            font-style: normal;
            border-radius: 0 4pt 4pt 0;
            page-break-inside: avoid;
        }}
        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 16pt 0;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_doc)

print(f"HTML generated at: {html_path}")

# Run headless Edge to produce PDF
edge_bin = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
cmd = [
    edge_bin,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_path}",
    str(html_path)
]

res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode == 0 and pdf_path.exists():
    print(f"SUCCESS: PDF created at {pdf_path} ({pdf_path.stat().st_size} bytes)")
else:
    print(f"FAILED: {res.stderr}")

