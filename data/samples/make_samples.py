"""Regenerates the two sample PDFs (original text, MIT-licensed with the project).

Requires: pip install reportlab
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).parent

DOCS = {
    "Artificial_Intelligence_Basics.pdf": [
        ("What is Artificial Intelligence?",
         "Artificial Intelligence (AI) is the field of computer science concerned with building systems that perform tasks "
         "normally requiring human intelligence, such as understanding language, recognising images and making decisions."),
        ("Machine Learning",
         "Machine Learning is a branch of AI that enables computers to learn patterns from data and make predictions "
         "without being explicitly programmed for each task. Supervised learning uses labelled examples, unsupervised "
         "learning finds structure in unlabelled data, and reinforcement learning learns from rewards."),
        ("Benefits of AI",
         "AI offers several benefits including increased efficiency through automation, better decision-making from data "
         "analysis, round-the-clock availability, and personalised experiences for users."),
        ("Risks and Limitations",
         "AI systems can inherit bias from their training data, may be difficult to interpret, and require careful "
         "oversight. Privacy and security of the data used to train them are important concerns."),
    ],
    "Digital_Marketing_SEO_Guide.pdf": [
        ("Introduction to SEO",
         "Search Engine Optimisation (SEO) is the practice of improving a website so that it appears higher in unpaid "
         "search engine results, increasing the quantity and quality of organic traffic."),
        ("On-Page SEO",
         "On-page SEO covers the elements you control on a page: title tags, meta descriptions, headings, keyword "
         "usage, internal links, image alt text and page speed."),
        ("Link Building",
         "Link building is the process of earning hyperlinks from other reputable websites. High-quality backlinks act "
         "as votes of confidence and help search engines judge the authority of a site."),
    ],
}

styles = getSampleStyleSheet()
for filename, pages in DOCS.items():
    doc = SimpleDocTemplate(str(OUT / filename), pagesize=A4, title=filename)
    story = []
    for i, (title, body) in enumerate(pages):
        story += [Paragraph(title, styles["Title"]), Spacer(1, 12)]
        story += [Paragraph(body, styles["BodyText"]), Spacer(1, 12)]
        if i < len(pages) - 1:
            story.append(PageBreak())
    doc.build(story)
    print("wrote", filename)
