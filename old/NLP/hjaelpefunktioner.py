"""Hjælpefunktioner til Natural Language Processing — KUN plotte-hjælp, ingen magi.

I behøver ikke læse koden her for at følge med. Men kig endelig: det er
helt almindelig matplotlib, som I kender — bare med lidt ekstra pynt.
Alt det, der er selve læringsmålet (data-imputation, text-normalization, text-encoding),
står synligt i notebooks'ene.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.style.use('ggplot') # MEGET VIGTIGT
from IPython.display import display, clear_output, HTML
import io, base64

def center_show(markdown_top='', markdown_bottom=''):
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()

    img = base64.b64encode(buf.getvalue()).decode()

    display(HTML(f"""
    <div style="max-width:800px; margin:auto;">
        <div>{markdown_top}</div>
        <br>
        <div style="text-align:center">
            <img src="data:image/png;base64,{img}">
        </div>
        <div>{markdown_bottom}</div>
    </div>
    """))

def plot_test_plot():
    markdown_top = '''
<h2 style='text-align: center;'><strong>Test plot med tekst ovenfor</strong></h2>
<div style='max-width: 100%; margin: 0 auto;'>
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since 1966, when designers at Letraset and James Mosley, the librarian at St Bride Printing Library in London, took a 1914 Cicero translation and scrambled it to make dummy text for Letraset's Body Type sheets. It has survived not only many decades, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised thanks to MIM.
</div>
'''

    markdown_bottom = '''
<h3 style='text-align: center;'><strong>Test plot med tekst ovenfor</strong></h3>
<div style='max-width: 100%; margin: 0 auto;'>
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since 1966, when designers at Letraset and James Mosley, the librarian at St Bride Printing Library in London, took a 1914 Cicero translation and scrambled it to make dummy text for Letraset's Body Type sheets. It has survived not only many decades, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised thanks to MIM.
</div>
'''
    
    center_show(markdown_top=markdown_top, markdown_bottom=markdown_bottom)