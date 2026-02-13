"""
Visualization: HTML Reports.

Aggregates all analysis plots and metrics into a single HTML dashboard
for easy sharing and review.
"""

from typing import Dict, List, Optional
import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt

class ReportGenerator:
    """
    Generates a static HTML report from plots and metrics.
    """

    def __init__(self, title: str = "Erasus Unlearning Report"):
        self.title = title
        self.sections = []
        self.metrics = {}

    def add_plot(self, fig: plt.Figure, title: str, description: str = ""):
        """
        Convert matplotlib figure to base64 image and add to report.
        """
        img_buf = BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        
        self.sections.append({
            "type": "plot",
            "title": title,
            "description": description,
            "content": f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;">'
        })
        plt.close(fig)

    def add_metrics(self, metrics: Dict[str, float]):
        """
        Add a table of metrics.
        """
        self.metrics.update(metrics)

    def save(self, filepath: str):
        """
        Render and save the HTML file.
        """
        html_content = self._render_html()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Report saved to {filepath}")

    def _render_html(self) -> str:
        # Simple CSS
        css = """
        body { font-family: sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; color: #333; }
        h1 { border-bottom: 2px solid #eee; padding-bottom: 10px; }
        .metric-card { background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .section { margin-bottom: 40px; }
        .plot-container { text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 10px; border-radius: 8px; }
        .description { font-style: italic; color: #666; margin-top: 10px; }
        """
        
        # Metrics Table
        metrics_html = ""
        if self.metrics:
            rows = "".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in self.metrics.items())
            metrics_html = f"""
            <div class="metric-card">
                <h2>Key Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    {rows}
                </table>
            </div>
            """

        # Sections
        sections_html = ""
        for section in self.sections:
            sections_html += f"""
            <div class="section">
                <h2>{section['title']}</h2>
                <div class="plot-container">
                    {section['content']}
                </div>
                <p class="description">{section['description']}</p>
            </div>
            """

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.title}</title>
            <style>{css}</style>
        </head>
        <body>
            <h1>{self.title}</h1>
            {metrics_html}
            {sections_html}
            <footer>Generared by Erasus Framework</footer>
        </body>
        </html>
        """
